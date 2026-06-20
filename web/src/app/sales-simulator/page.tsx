"use client";

import { useMemo, useState } from "react";
import Link from "next/link";
import {
  calculateSalesSimulation,
  DEFAULT_SALES_INPUTS,
  type SalesInputs,
} from "../../lib/salesSimulation";

const GBP = new Intl.NumberFormat("en-GB", {
  style: "currency",
  currency: "GBP",
  maximumFractionDigits: 0,
});
const NUMBER = new Intl.NumberFormat("en-GB", { maximumFractionDigits: 1 });

type NumericKey = keyof SalesInputs;

const conversionInputs: SliderConfig[] = [
  { key: "targetAccounts", label: "Target accounts", min: 10, max: 250, step: 5, suffix: "" },
  { key: "qualifiedRate", label: "Qualified conversation rate", min: 0, max: 100, step: 1, suffix: "%" },
  { key: "designPartnerRate", label: "Design partner conversion", min: 0, max: 100, step: 1, suffix: "%" },
  { key: "paidPilotRate", label: "Paid pilot conversion", min: 0, max: 100, step: 1, suffix: "%" },
  { key: "membershipRate", label: "Pilot-to-membership conversion", min: 0, max: 100, step: 1, suffix: "%" },
];

const commercialInputs: SliderConfig[] = [
  { key: "paidPilotPrice", label: "Paid pilot price", min: 0, max: 150_000, step: 5_000, prefix: "£" },
  { key: "annualMembershipPrice", label: "Annual membership", min: 0, max: 500_000, step: 10_000, prefix: "£" },
  { key: "intelligenceModulePrice", label: "Intelligence module", min: 0, max: 200_000, step: 5_000, prefix: "£" },
  { key: "moduleAttachRate", label: "Module attach rate", min: 0, max: 100, step: 1, suffix: "%" },
];

const costInputs: SliderConfig[] = [
  { key: "pilotDeliveryCost", label: "Delivery cost per pilot", min: 0, max: 120_000, step: 5_000, prefix: "£" },
  { key: "annualServiceCost", label: "Annual cost per member", min: 0, max: 150_000, step: 5_000, prefix: "£" },
  { key: "fixedProgrammeCost", label: "Fixed programme cost", min: 0, max: 1_000_000, step: 25_000, prefix: "£" },
];

export default function SalesSimulatorPage() {
  const [inputs, setInputs] = useState<SalesInputs>(DEFAULT_SALES_INPUTS);
  const result = useMemo(() => calculateSalesSimulation(inputs), [inputs]);

  const update = (key: NumericKey, value: number) => {
    setInputs((current) => ({ ...current, [key]: Number.isFinite(value) ? value : 0 }));
  };

  const pipeline = [
    { label: "Target accounts", value: inputs.targetAccounts, accent: false },
    { label: "Qualified calls", value: result.qualifiedCalls, accent: false },
    { label: "Design partners", value: result.designPartners, accent: false },
    { label: "Paid pilots", value: result.paidPilots, accent: false },
    { label: "Network members", value: result.networkMembers, accent: true },
  ];

  return (
    <main className="mx-auto flex min-h-screen w-full max-w-[1500px] flex-col gap-6 px-5 py-7 sm:px-8 sm:py-10">
      <header className="flex flex-col gap-4">
        <div className="flex flex-wrap items-start justify-between gap-5">
          <div>
            <Link href="/" className="eyebrow transition-colors hover:text-accent-gold">
              Veritas / live fraud demo
            </Link>
            <h1 className="mt-3 font-display text-[clamp(2.2rem,6vw,4.4rem)] leading-[0.95] tracking-tight text-text-primary">
              Sales pipeline simulator
            </h1>
            <p className="mt-4 max-w-3xl text-[15px] leading-relaxed text-text-secondary sm:text-base">
              Change conversion, pricing and delivery assumptions to see how a founding network
              could translate into members, revenue and profit.
            </p>
          </div>
          <div className="rounded-full border border-accent-gold/40 bg-accent-gold/10 px-4 py-2 text-[11px] font-semibold uppercase tracking-[0.18em] text-accent-gold">
            Forecast, not traction
          </div>
        </div>
        <div className="hairline" />
      </header>

      <section className="grid gap-4 sm:grid-cols-2 xl:grid-cols-4">
        <MetricCard label="Expected network members" value={NUMBER.format(result.networkMembers)} detail={`${NUMBER.format(result.paidPilots)} paid pilots`} tone="green" />
        <MetricCard label="Modelled revenue" value={GBP.format(result.totalRevenue)} detail={`${GBP.format(result.membershipRevenue)} membership`} />
        <MetricCard label="Modelled profit" value={GBP.format(result.profit)} detail={`${(result.margin * 100).toFixed(1)}% margin`} tone={result.profit >= 0 ? "green" : "red"} />
        <MetricCard label="Break-even network" value={result.breakEvenMembers === null ? "Not reached" : `${result.breakEvenMembers} members`} detail={`${GBP.format(result.totalCost)} modelled cost`} />
      </section>

      <section className="rounded-[20px] border border-border-default bg-bg-surface/75 p-5 shadow-[var(--shadow-raise)] sm:p-7">
        <div className="flex flex-wrap items-end justify-between gap-3">
          <div>
            <p className="eyebrow text-fed">Expected pipeline</p>
            <h2 className="mt-2 font-display text-2xl tracking-tight text-text-primary">From account list to network</h2>
          </div>
          <p className="max-w-xl text-right text-[12px] leading-relaxed text-text-muted">
            Expected values may be fractional because the model applies conversion rates rather than claiming completed deals.
          </p>
        </div>
        <div className="mt-6 grid gap-3 md:grid-cols-5">
          {pipeline.map((stage, index) => {
            const width = inputs.targetAccounts > 0 ? Math.max(7, (stage.value / inputs.targetAccounts) * 100) : 7;
            return (
              <div key={stage.label} className="relative overflow-hidden rounded-[16px] border border-border-default bg-bg-deep p-4">
                <div className="absolute inset-y-0 left-0 opacity-15" style={{ width: `${width}%`, background: stage.accent ? "var(--fed)" : "var(--accent-gold)" }} />
                <div className="relative">
                  <p className="font-mono text-[10px] text-text-muted">0{index + 1}</p>
                  <p className="mt-4 tabular font-display text-3xl text-text-primary">{NUMBER.format(stage.value)}</p>
                  <p className="mt-1 text-[12px] font-semibold text-text-secondary">{stage.label}</p>
                </div>
              </div>
            );
          })}
        </div>
      </section>

      <div className="grid gap-5 xl:grid-cols-[1.05fr_1.45fr]">
        <section className="rounded-[20px] border border-border-default bg-bg-surface/75 p-5 shadow-[var(--shadow-raise)] sm:p-7">
          <div className="flex items-center justify-between gap-3">
            <div>
              <p className="eyebrow text-accent-gold">Model inputs</p>
              <h2 className="mt-2 font-display text-2xl text-text-primary">Tune the assumptions</h2>
            </div>
            <button type="button" onClick={() => setInputs(DEFAULT_SALES_INPUTS)} className="rounded-[10px] border border-border-strong px-3 py-2 text-[12px] font-semibold text-text-secondary transition-colors hover:bg-bg-surface-2 hover:text-text-primary">
              Reset model
            </button>
          </div>

          <InputGroup title="Acquisition and conversion" configs={conversionInputs} inputs={inputs} update={update} />
          <InputGroup title="Commercial model" configs={commercialInputs} inputs={inputs} update={update} />
          <InputGroup title="Delivery costs" configs={costInputs} inputs={inputs} update={update} />
        </section>

        <section className="flex flex-col gap-5">
          <RevenuePanel result={result} />
          <AssumptionsPanel />
        </section>
      </div>

      <footer className="pb-4 pt-2 text-[12px] leading-relaxed text-text-muted">
        This tool is an assumption model. It does not represent contracted customers, validated pricing or forecast revenue.
      </footer>
    </main>
  );
}

interface SliderConfig {
  key: NumericKey;
  label: string;
  min: number;
  max: number;
  step: number;
  prefix?: string;
  suffix?: string;
}

function InputGroup({ title, configs, inputs, update }: { title: string; configs: SliderConfig[]; inputs: SalesInputs; update: (key: NumericKey, value: number) => void }) {
  return (
    <div className="mt-7 border-t border-border-default pt-5">
      <p className="text-[12px] font-semibold uppercase tracking-[0.16em] text-text-muted">{title}</p>
      <div className="mt-4 space-y-5">
        {configs.map((config) => (
          <label key={config.key} className="block">
            <span className="flex items-center justify-between gap-4 text-[13px]">
              <span className="text-text-secondary">{config.label}</span>
              <span className="tabular font-mono text-text-primary">
                {config.prefix}{NUMBER.format(inputs[config.key])}{config.suffix}
              </span>
            </span>
            <input
              type="range"
              min={config.min}
              max={config.max}
              step={config.step}
              value={inputs[config.key]}
              onChange={(event) => update(config.key, Number(event.target.value))}
              className="mt-2 h-1.5 w-full cursor-pointer appearance-none rounded-full bg-border-strong accent-[#d6a85b]"
            />
          </label>
        ))}
      </div>
    </div>
  );
}

function MetricCard({ label, value, detail, tone }: { label: string; value: string; detail: string; tone?: "green" | "red" }) {
  const color = tone === "green" ? "var(--fed)" : tone === "red" ? "var(--silo)" : "var(--text-primary)";
  return (
    <article className="rounded-[18px] border border-border-default bg-bg-surface/75 p-5 shadow-[var(--shadow-raise)]">
      <p className="eyebrow">{label}</p>
      <p className="mt-4 tabular font-display text-[clamp(1.8rem,4vw,2.6rem)] leading-none tracking-tight" style={{ color }}>{value}</p>
      <p className="mt-3 text-[12px] text-text-muted">{detail}</p>
    </article>
  );
}

function RevenuePanel({ result }: { result: ReturnType<typeof calculateSalesSimulation> }) {
  const rows = [
    { label: "Paid pilots", value: result.pilotRevenue, color: "var(--accent-gold)" },
    { label: "Annual memberships", value: result.membershipRevenue, color: "var(--fed)" },
    { label: "Intelligence modules", value: result.moduleRevenue, color: "var(--accent-gold-soft)" },
  ];
  const scale = Math.max(result.totalRevenue, result.totalCost, 1);

  return (
    <div className="rounded-[20px] border border-border-default bg-bg-surface/75 p-5 shadow-[var(--shadow-raise)] sm:p-7">
      <p className="eyebrow text-fed">Economics</p>
      <h2 className="mt-2 font-display text-2xl text-text-primary">Revenue composition</h2>
      <div className="mt-7 space-y-5">
        {rows.map((row) => (
          <div key={row.label}>
            <div className="flex items-center justify-between gap-3 text-[13px]">
              <span className="text-text-secondary">{row.label}</span>
              <span className="tabular font-mono text-text-primary">{GBP.format(row.value)}</span>
            </div>
            <div className="mt-2 h-2 overflow-hidden rounded-full bg-bg-deep">
              <div className="h-full rounded-full transition-[width] duration-300" style={{ width: `${Math.max(0, (row.value / scale) * 100)}%`, background: row.color }} />
            </div>
          </div>
        ))}
      </div>
      <div className="mt-7 grid grid-cols-2 gap-3 border-t border-border-default pt-5">
        <ResultLine label="Total revenue" value={GBP.format(result.totalRevenue)} />
        <ResultLine label="Total cost" value={GBP.format(result.totalCost)} />
        <ResultLine label="Module customers" value={NUMBER.format(result.moduleCustomers)} />
        <ResultLine label="Net profit" value={GBP.format(result.profit)} highlight={result.profit >= 0} />
      </div>
    </div>
  );
}

function ResultLine({ label, value, highlight }: { label: string; value: string; highlight?: boolean }) {
  return (
    <div className="rounded-[14px] bg-bg-deep p-4">
      <p className="text-[11px] text-text-muted">{label}</p>
      <p className="mt-2 tabular font-mono text-[15px] font-semibold" style={{ color: highlight ? "var(--fed)" : "var(--text-primary)" }}>{value}</p>
    </div>
  );
}

function AssumptionsPanel() {
  return (
    <div className="rounded-[20px] border border-accent-gold/30 bg-accent-gold/5 p-5 sm:p-7">
      <p className="eyebrow text-accent-gold">How to read this</p>
      <div className="mt-4 grid gap-4 text-[13px] leading-relaxed text-text-secondary sm:grid-cols-3">
        <p><strong className="text-text-primary">Expected values</strong><br />Conversion rates produce mathematical expectations, not whole signed customers.</p>
        <p><strong className="text-text-primary">One-year view</strong><br />Membership and service costs represent a single annual period.</p>
        <p><strong className="text-text-primary">No hidden confidence</strong><br />Inputs remain hypotheses until interviews and paid pilots provide evidence.</p>
      </div>
    </div>
  );
}
