export type ScamTypeData = {
  type: string;
  count: number;
  color: string;
};

export type ScamLocation = {
  city: string;
  country: string;
  lat: number;
  lng: number;
  scamCount: number;
  scamTypes: ScamTypeData[];
  risk: number; // 0-1 risk value
};

export type GlobalStats = {
  totalScams: number;
  avgRisk: number;
  scamTypeBreakdown: {
    type: string;
    count: number;
    percentage: number;
  }[];
  highRiskLocations: ScamLocation[];
};