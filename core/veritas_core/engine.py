import numpy as np
from .data import make_bank_data, inject_campaign, FEATURE_DIM
from .model import init_weights, train_local, recall
from .aggregation import fedavg
from .dp import privatize
from .attack import poisoned_update

NAMES=["Barclays","NatWest","Lloyds","HSBC","Santander","Monzo","Starling","Nationwide"]
CUST=[2_100_000,1_900_000,1_750_000,1_600_000,1_400_000,900_000,700_000,1_500_000]
AVG_LOSS=255; AT_RISK=1500; THRESH=0.9; HOURS=1.0; MAX_NORM=2.0; SIGMA=0.05; EPOCHS=8
# Federated recall the global model must regain before we surface the
# "malicious member rejected" beat, proving the defence preserved the model.
ATTACK_RECOVERED=0.75


def _krum_scores(U, n_byzantine):
    """Per-update Krum score: sum of squared dists to the k nearest peers.
    A poisoned (sign-flipped, amplified) update sits far from the honest cloud
    and therefore earns the largest score."""
    n = len(U); k = max(1, n - n_byzantine - 2)
    dist = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            d = float(np.sum((U[i] - U[j]) ** 2)); dist[i, j] = dist[j, i] = d
    return np.array([np.sort(dist[i])[1:k + 1].sum() for i in range(n)])


class Engine:
    def __init__(self, n_banks=8, seed=0, use_dp=True, use_krum=True):
        self.n = n_banks; self.rng = np.random.default_rng(seed)
        self.use_dp = use_dp; self.use_krum = use_krum
        self.round = 0; self.campaign = False; self.attack_bank = None
        self.global_w = init_weights(FEATURE_DIM)
        self.silo_w = [init_weights(FEATURE_DIM) for _ in range(n_banks)]
        self.data = [make_bank_data(3000, 0.03, seed=seed + i) for i in range(n_banks)]
        self.eval = [make_bank_data(1000, 0.05, seed=seed + 100 + i) for i in range(n_banks)]
        # banks whose LOCAL training data carries the campaign (federated learns
        # it from these and protects everyone; siloed banks without it stay blind)
        self.seen = [False] * n_banks
        self.cum = {"fed": 0.0, "silo": 0.0}
        self._rejected_ever = False; self._attack_announced = False

    def inject_campaign(self):
        """Start the scam campaign: every bank's customers are now TARGETED
        (campaign frauds appear in every eval set), but only a subset of banks
        have campaign examples in their local training data."""
        self.campaign = True
        n_seed = max(1, self.n // 2)
        for i in range(self.n):
            self.eval[i] = inject_campaign(*self.eval[i], 80, seed=200 + i)
        for b in range(n_seed):
            self.data[b] = inject_campaign(*self.data[b], 150, seed=1 + b)
            self.seen[b] = True

    def inject_attack(self, bank_id):
        self.attack_bank = int(bank_id.replace("bank", "")) if bank_id.startswith("bank") else 0

    def _aggregate(self, ups):
        """Federated aggregation. With no attack: FedAvg over all honest updates
        (keeps every campaign-learning signal). Under attack: use Krum scoring to
        drop the single most-outlying update, then FedAvg the survivors so the
        poison is filtered without discarding honest campaign knowledge.
        Returns (aggregate, rejected_index_or_None)."""
        U = np.stack(ups); rejected = None
        if self.use_krum and self.attack_bank is not None:
            worst = int(np.argmax(_krum_scores(U, n_byzantine=1)))
            keep = [i for i in range(self.n) if i != worst]
            rejected = worst
            return fedavg([U[i] for i in keep]), rejected
        return fedavg(ups), rejected

    def step(self):
        ev = []; self.round += 1
        ups = []
        for i in range(self.n):
            X, y = self.data[i]
            wl = train_local(self.global_w, X, y, epochs=EPOCHS, lr=0.3)
            u = wl - self.global_w
            if self.use_dp:
                u = privatize(u, MAX_NORM, SIGMA, self.rng)
            ups.append(u)
        if self.attack_bank is not None and self.round < 3:
            ups[self.attack_bank] = poisoned_update(ups[self.attack_bank], 12.0)
        agg, rejected = self._aggregate(ups)
        if self.attack_bank is not None and rejected == self.attack_bank:
            self._rejected_ever = True
        self.global_w = self.global_w + agg
        for i in range(self.n):
            X, y = self.data[i]
            self.silo_w[i] = train_local(self.silo_w[i], X, y, epochs=EPOCHS, lr=0.3)
        fed, silo = self._det()
        # Announce the rejected malicious member only once its updates have been
        # filtered AND the global model has demonstrably recovered, so the demo's
        # "attack rejected -> model still healthy" beat lands on a protected state.
        if (self.attack_bank is not None and self._rejected_ever
                and not self._attack_announced and max(fed) > ATTACK_RECOVERED):
            self._attack_announced = True
            ev.append({"type": "attack_detected",
                       "data": {"bankId": f"bank{self.attack_bank}", "rejected": True}})
        for i in range(self.n):
            ev.append({"type": "client_updated", "data": {"bankId": f"bank{i}",
                       "detection": {"federated": fed[i], "siloed": silo[i]}}})
        self.cum["fed"] += sum(AT_RISK * (1 - d) for d in fed) / self.n
        self.cum["silo"] += sum(AT_RISK * (1 - d) for d in silo) / self.n
        ev.append({"type": "round_complete", "data": self.state()})
        return ev

    def _det(self):
        fed = [recall(self.global_w, self.eval[i][0], self.eval[i][1]) for i in range(self.n)]
        silo = [recall(self.silo_w[i], self.eval[i][0], self.eval[i][1]) for i in range(self.n)]
        return fed, silo

    def state(self):
        fed, silo = self._det()
        banks = [{"id": f"bank{i}", "name": NAMES[i], "customers": CUST[i],
                  "detection": {"federated": round(fed[i], 3), "siloed": round(silo[i], 3)},
                  "poisoned": self.attack_bank == i and self.round < 3} for i in range(self.n)]
        fv = int(self.cum["fed"]); sv = int(self.cum["silo"])
        ttd = lambda a: HOURS * self.round if a >= THRESH else 101.0
        return {"round": self.round, "running": True, "banks": banks,
                "campaignActive": self.campaign, "attackActive": self.attack_bank is not None,
                "customerRecordsTransmitted": 0,
                "counters": {
                    "federated": {"fraudPreventedGbp": max(0, sv - fv) * AVG_LOSS,
                                  "timeToDetectHours": ttd(sum(fed) / self.n),
                                  "victims": fv, "lostGbp": fv * AVG_LOSS},
                    "siloed": {"fraudPreventedGbp": 0,
                               "timeToDetectHours": ttd(sum(silo) / self.n),
                               "victims": sv, "lostGbp": sv * AVG_LOSS}}}
