import type { Metadata } from "next";
import { SalesSimulator } from "../../features/sales-simulator";

export const metadata: Metadata = {
  title: "Sales Pipeline Simulator | Veritas",
  description:
    "Explore how conversion, pricing and delivery assumptions affect the Veritas commercial pathway.",
};

export default function SalesSimulatorPage() {
  return <SalesSimulator />;
}
