import EmbeddingAtlas from "./components/EmbeddingAtlas";
import EnsembleStack from "./components/EnsembleStack";
import MuleGraph from "./components/MuleGraph";
import SequenceTimeline from "./components/SequenceTimeline";
import FederatedGbdt from "./components/FederatedGbdt";
import FederationPulse from "./components/FederationPulse";
import PrivacyBudget from "./components/PrivacyBudget";
import ZkProof from "./components/ZkProof";

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

      <main className="mx-auto w-full max-w-[1100px] px-5 sm:px-8 py-8 flex flex-col gap-6 [&_canvas]:max-w-full">
        <EmbeddingAtlas />
        <EnsembleStack />
        <MuleGraph />
        <SequenceTimeline />
        <FederatedGbdt />
        <FederationPulse />
        <PrivacyBudget />
        <ZkProof />
      </main>

      <footer className="border-t border-border-default">
        <div className="mx-auto max-w-[1180px] px-5 sm:px-8 py-6 text-text-muted text-xs">
          Veritas · Static technology showcase — fully client-side, no backend.
        </div>
      </footer>
    </div>
  );
}
