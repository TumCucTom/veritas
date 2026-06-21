import type { ReactNode } from "react";
import EmbeddingAtlas from "./components/EmbeddingAtlas";
import EnsembleStack from "./components/EnsembleStack";
import MuleGraph from "./components/MuleGraph";
import SequenceTimeline from "./components/SequenceTimeline";
import FederatedGbdt from "./components/FederatedGbdt";
import FederationPulse from "./components/FederationPulse";
import PrivacyBudget from "./components/PrivacyBudget";
import ZkProof from "./components/ZkProof";

function Card({
  eyebrow,
  title,
  children,
}: {
  eyebrow: string;
  title: string;
  children: ReactNode;
}) {
  return (
    <section
      className="rounded-[var(--radius-panel)] border border-border-default bg-bg-surface p-5 sm:p-6"
      style={{ boxShadow: "var(--shadow-panel)" }}
    >
      <div className="mb-4">
        <div className="eyebrow">{eyebrow}</div>
        <h2 className="font-display text-lg sm:text-xl text-text-primary mt-1">
          {title}
        </h2>
      </div>
      {children}
    </section>
  );
}

export default function App() {
  return (
    <div className="min-h-full">
      <header className="border-b border-border-default">
        <div className="mx-auto max-w-[1180px] px-5 sm:px-8 py-5 flex flex-col gap-3 sm:flex-row sm:items-end sm:justify-between">
          <div>
            <div className="eyebrow">Veritas</div>
            <h1 className="font-display text-2xl sm:text-3xl text-text-primary mt-1">
              Under the hood — live from the models
            </h1>
            <p className="text-text-secondary text-sm mt-2 max-w-[640px]">
              Every panel is precomputed from the real Veritas models, running
              entirely in your browser.
            </p>
          </div>
          <a
            href="../index.html"
            className="text-sm text-accent-gold hover:text-text-primary transition-colors whitespace-nowrap"
          >
            ← Back to Veritas
          </a>
        </div>
      </header>

      <main className="mx-auto max-w-[1180px] px-5 sm:px-8 py-8 flex flex-col gap-6">
        <Card
          eyebrow="Representation"
          title="Embedding Atlas — learned fraud manifold"
        >
          <EmbeddingAtlas />
        </Card>

        <Card eyebrow="Decisioning" title="Ensemble Stack — model vote fusion">
          <EnsembleStack />
        </Card>

        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
          <Card eyebrow="Graph" title="Mule Graph — laundering rings">
            <MuleGraph />
          </Card>
          <Card
            eyebrow="Temporal"
            title="Sequence Timeline — transaction signals"
          >
            <SequenceTimeline />
          </Card>
          <Card
            eyebrow="Federation"
            title="Federated GBDT — split-learning trees"
          >
            <FederatedGbdt />
          </Card>
          <Card
            eyebrow="Federation"
            title="Federation Pulse — secure aggregation"
          >
            <FederationPulse />
          </Card>
          <Card eyebrow="Privacy" title="Privacy Budget — differential privacy">
            <PrivacyBudget />
          </Card>
          <Card eyebrow="Verification" title="ZK Proof — cryptographic attest">
            <ZkProof />
          </Card>
        </div>
      </main>

      <footer className="border-t border-border-default">
        <div className="mx-auto max-w-[1180px] px-5 sm:px-8 py-6 text-text-muted text-xs">
          Veritas · Static technology showcase — fully client-side, no backend.
        </div>
      </footer>
    </div>
  );
}
