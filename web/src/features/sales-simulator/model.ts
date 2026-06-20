export interface SalesInputs {
  targetAccounts: number;
  qualifiedRate: number;
  designPartnerRate: number;
  paidPilotRate: number;
  membershipRate: number;
  paidPilotPrice: number;
  annualMembershipPrice: number;
  intelligenceModulePrice: number;
  moduleAttachRate: number;
  pilotDeliveryCost: number;
  annualServiceCost: number;
  fixedProgrammeCost: number;
}

export interface SalesSimulation {
  qualifiedCalls: number;
  designPartners: number;
  paidPilots: number;
  networkMembers: number;
  moduleCustomers: number;
  pilotRevenue: number;
  membershipRevenue: number;
  moduleRevenue: number;
  totalRevenue: number;
  totalCost: number;
  profit: number;
  margin: number;
  breakEvenMembers: number | null;
}

export const DEFAULT_SALES_INPUTS: SalesInputs = {
  targetAccounts: 100,
  qualifiedRate: 30,
  designPartnerRate: 33,
  paidPilotRate: 50,
  membershipRate: 60,
  paidPilotPrice: 50_000,
  annualMembershipPrice: 150_000,
  intelligenceModulePrice: 50_000,
  moduleAttachRate: 40,
  pilotDeliveryCost: 30_000,
  annualServiceCost: 40_000,
  fixedProgrammeCost: 250_000,
};

const asRate = (value: number): number => Math.min(100, Math.max(0, value)) / 100;

export function calculateSalesSimulation(inputs: SalesInputs): SalesSimulation {
  const qualifiedCalls = inputs.targetAccounts * asRate(inputs.qualifiedRate);
  const designPartners = qualifiedCalls * asRate(inputs.designPartnerRate);
  const paidPilots = designPartners * asRate(inputs.paidPilotRate);
  const networkMembers = paidPilots * asRate(inputs.membershipRate);
  const moduleCustomers = networkMembers * asRate(inputs.moduleAttachRate);

  const pilotRevenue = paidPilots * inputs.paidPilotPrice;
  const membershipRevenue = networkMembers * inputs.annualMembershipPrice;
  const moduleRevenue = moduleCustomers * inputs.intelligenceModulePrice;
  const totalRevenue = pilotRevenue + membershipRevenue + moduleRevenue;

  const totalCost =
    inputs.fixedProgrammeCost +
    paidPilots * inputs.pilotDeliveryCost +
    networkMembers * inputs.annualServiceCost;
  const profit = totalRevenue - totalCost;
  const margin = totalRevenue > 0 ? profit / totalRevenue : 0;

  const annualContributionPerMember =
    inputs.annualMembershipPrice +
    asRate(inputs.moduleAttachRate) * inputs.intelligenceModulePrice -
    inputs.annualServiceCost;
  const netProgrammeAndPilotCost =
    inputs.fixedProgrammeCost + paidPilots * (inputs.pilotDeliveryCost - inputs.paidPilotPrice);
  const breakEvenMembers =
    annualContributionPerMember > 0
      ? Math.max(0, Math.ceil(netProgrammeAndPilotCost / annualContributionPerMember))
      : null;

  return {
    qualifiedCalls,
    designPartners,
    paidPilots,
    networkMembers,
    moduleCustomers,
    pilotRevenue,
    membershipRevenue,
    moduleRevenue,
    totalRevenue,
    totalCost,
    profit,
    margin,
    breakEvenMembers,
  };
}
