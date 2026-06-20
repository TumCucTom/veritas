import { VeritasProvider } from "../lib/store";
import { ControlPlaneProvider } from "../lib/controlPlaneStore";
import Controls from "../components/Controls";
import HeroCounters from "../components/HeroCounters";
import MassiveContagionStage from "../components/MassiveContagionStage";
import TechnicalProofStrip from "../components/TechnicalProofStrip";
import RacePanel from "../components/RacePanel";
import BankStrip from "../components/BankStrip";
import Inspector from "../components/Inspector";
import PrivacyBadge from "../components/PrivacyBadge";
import AttackBanner from "../components/AttackBanner";
import ProvenancePanel from "../components/ProvenancePanel";
import GovernancePanel from "../components/GovernancePanel";
import EdgeFleetPanel from "../components/EdgeFleetPanel";

export default function Home() {
  return (
    // Two coexisting data sources: VeritasProvider reads the bank NODE (Tier 1,
    // NEXT_PUBLIC_API_BASE) for the live race / hero / inspector;
    // ControlPlaneProvider reads the federation CONTROL PLANE (Tier 2,
    // NEXT_PUBLIC_CONTROL_PLANE) for governance / provenance / edge fleet.
    <VeritasProvider>
      <ControlPlaneProvider>
        <div className="mx-auto flex min-h-screen w-full max-w-[1400px] flex-col gap-6 px-5 py-7 sm:px-8 sm:py-10">
          <SiteHeader />

          <Controls />
          <AttackBanner />
          <HeroCounters />
          <TechnicalProofStrip />
          <MassiveContagionStage />

          <div className="grid grid-cols-1 gap-5 lg:grid-cols-2">
            <RacePanel regime="siloed" />
            <RacePanel regime="federated" />
          </div>

          <BankStrip />

          <GovernancePanel />

          <div className="grid grid-cols-1 gap-5 lg:grid-cols-[1.4fr_1fr]">
            <Manifesto />
            <Inspector />
          </div>

          <div className="grid grid-cols-1 gap-5 lg:grid-cols-2">
            <ProvenancePanel />
            <EdgeFleetPanel />
          </div>

          <SiteFooter />
        </div>
      </ControlPlaneProvider>
    </VeritasProvider>
  );
}

function SiteHeader() {
  return (
    <header className="flex flex-col gap-4">
      <div className="flex flex-wrap items-center justify-between gap-4">
        <div className="flex items-center gap-3">
          <span
            aria-hidden
            className="grid h-9 w-9 place-items-center rounded-[10px] font-display text-lg font-semibold text-bg-deep"
            style={{
              background: "linear-gradient(150deg, var(--accent-gold-soft), var(--accent-gold))",
              boxShadow: "0 6px 18px -8px rgba(214,168,91,0.8)",
            }}
          >
            V
          </span>
          <div>
            <h1 className="font-display text-2xl leading-none tracking-tight text-text-primary">
              Veritas
            </h1>
            <p className="mt-1 text-[12px] tracking-wide text-text-muted">
              Federated fraud intelligence · FLock.io × UK Sovereign AI
            </p>
          </div>
        </div>
        <PrivacyBadge />
      </div>
      <div className="hairline" />
    </header>
  );
}

function Manifesto() {
  return (
    <section
      className="flex flex-col justify-center rounded-[20px] border bg-bg-surface/70 p-6 backdrop-blur-sm sm:p-8"
      style={{ borderColor: "var(--border-default)", boxShadow: "var(--shadow-raise)" }}
    >
      <p className="eyebrow text-accent-gold">The sovereignty case</p>
      <p className="mt-3 max-w-xl font-display text-[clamp(1.25rem,1rem+1vw,1.75rem)] leading-snug tracking-tight text-text-primary">
        Fraud crosses banks in minutes. Intelligence shouldn&apos;t take{" "}
        <span style={{ color: "var(--silo)" }}>weeks</span>.
      </p>
      <p className="mt-4 max-w-xl text-[14px] leading-relaxed text-text-secondary">
        Veritas trains one model across every institution&apos;s books without a single customer
        record leaving the bank. Each member shares only privacy-clipped, noise-protected gradients;
        Multi-Krum rejects any poisoned contribution. The result is collective immunity to a scam
        campaign — under UK data sovereignty, on infrastructure the institutions own.
      </p>
    </section>
  );
}

function SiteFooter() {
  return (
    <footer className="mt-2 flex flex-col gap-1 pt-2">
      <div className="hairline mb-4" />
      <p className="text-[12px] text-text-muted">
        Differential privacy + Multi-Krum aggregation · production path anchored on FLock on-chain
        attestation. Figures are simulated for demonstration.
      </p>
    </footer>
  );
}
