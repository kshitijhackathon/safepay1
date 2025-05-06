import { ScamLocation } from '@/types/fraud-map';

// Default scam types with consistent colors
export const SCAM_TYPES = [
  { type: 'Phishing', color: '#FF5252' },
  { type: 'Identity Theft', color: '#FF7043' },
  { type: 'UPI Fraud', color: '#FFCA28' },
  { type: 'Investment Scam', color: '#66BB6A' },
  { type: 'Romance Scam', color: '#42A5F5' },
  { type: 'Job Scam', color: '#AB47BC' },
  { type: 'Banking Fraud', color: '#EC407A' },
  { type: 'Tech Support Scam', color: '#26A69A' },
  { type: 'Online Shopping', color: '#5C6BC0' },
  { type: 'Government Impersonation', color: '#8D6E63' }
];

// Sample data generator for development/demo purposes
export function getScamLocationData(): Promise<ScamLocation[]> {
  // In a real implementation, fetch from API:
  // return fetch('/api/fraud-map/locations').then(res => res.json());
  
  // For now, generate sample data across India
  const majorCities = [
    { city: 'Mumbai', state: 'Maharashtra', lat: 19.076, lng: 72.877, risk: 0.85 },
    { city: 'Delhi', state: 'Delhi', lat: 28.704, lng: 77.102, risk: 0.78 },
    { city: 'Bangalore', state: 'Karnataka', lat: 12.972, lng: 77.594, risk: 0.65 },
    { city: 'Hyderabad', state: 'Telangana', lat: 17.385, lng: 78.487, risk: 0.72 },
    { city: 'Chennai', state: 'Tamil Nadu', lat: 13.083, lng: 80.270, risk: 0.58 },
    { city: 'Kolkata', state: 'West Bengal', lat: 22.563, lng: 88.363, risk: 0.82 },
    { city: 'Ahmedabad', state: 'Gujarat', lat: 23.033, lng: 72.586, risk: 0.61 },
    { city: 'Pune', state: 'Maharashtra', lat: 18.520, lng: 73.856, risk: 0.55 },
    { city: 'Jaipur', state: 'Rajasthan', lat: 26.922, lng: 75.778, risk: 0.69 },
    { city: 'Lucknow', state: 'Uttar Pradesh', lat: 26.847, lng: 80.947, risk: 0.71 },
    { city: 'Chandigarh', state: 'Punjab', lat: 30.734, lng: 76.779, risk: 0.43 },
    { city: 'Bhopal', state: 'Madhya Pradesh', lat: 23.259, lng: 77.413, risk: 0.48 },
    { city: 'Patna', state: 'Bihar', lat: 25.596, lng: 85.138, risk: 0.76 },
    { city: 'Kochi', state: 'Kerala', lat: 9.939, lng: 76.270, risk: 0.39 },
    { city: 'Guwahati', state: 'Assam', lat: 26.144, lng: 91.736, risk: 0.51 },
    { city: 'Bhubaneswar', state: 'Odisha', lat: 20.296, lng: 85.824, risk: 0.47 },
    { city: 'Dehradun', state: 'Uttarakhand', lat: 30.316, lng: 78.032, risk: 0.41 },
    { city: 'Raipur', state: 'Chhattisgarh', lat: 21.250, lng: 81.630, risk: 0.53 },
    { city: 'Ranchi', state: 'Jharkhand', lat: 23.344, lng: 85.309, risk: 0.62 },
    { city: 'Shimla', state: 'Himachal Pradesh', lat: 31.104, lng: 77.173, risk: 0.29 },
    { city: 'Srinagar', state: 'Jammu & Kashmir', lat: 34.084, lng: 74.797, risk: 0.58 },
    { city: 'Thiruvananthapuram', state: 'Kerala', lat: 8.524, lng: 76.936, risk: 0.35 },
    // International cities for global perspective  
    { city: 'New York', state: 'USA', lat: 40.713, lng: -74.006, risk: 0.67 },
    { city: 'London', state: 'UK', lat: 51.507, lng: -0.127, risk: 0.59 },
    { city: 'Tokyo', state: 'Japan', lat: 35.682, lng: 139.759, risk: 0.42 },
    { city: 'Sydney', state: 'Australia', lat: -33.868, lng: 151.209, risk: 0.47 },
    { city: 'Singapore', state: 'Singapore', lat: 1.352, lng: 103.820, risk: 0.31 },
    { city: 'Dubai', state: 'UAE', lat: 25.204, lng: 55.270, risk: 0.64 },
    { city: 'Nairobi', state: 'Kenya', lat: -1.286, lng: 36.817, risk: 0.73 },
    { city: 'Lagos', state: 'Nigeria', lat: 6.455, lng: 3.396, risk: 0.81 }
  ];
  
  return new Promise(resolve => {
    // Convert to ScamLocation format with randomly generated data
    const locations: ScamLocation[] = majorCities.map(city => {
      // Generate a realistic number of scams based on risk level and randomization
      const baseScamCount = Math.floor(city.risk * 1000);
      const randomVariation = Math.floor(Math.random() * 200) - 100; // -100 to +100
      const scamCount = Math.max(50, baseScamCount + randomVariation);
      
      // Generate scam type breakdown
      // Select 4-7 random scam types for this location
      const numScamTypes = 4 + Math.floor(Math.random() * 4);
      const selectedTypes = [...SCAM_TYPES]
        .sort(() => 0.5 - Math.random())
        .slice(0, numScamTypes);
      
      // Distribute scam count across selected types
      let remainingCount = scamCount;
      const scamTypes = selectedTypes.map((type, index, array) => {
        // Last item gets remaining count
        if (index === array.length - 1) {
          return { ...type, count: remainingCount };
        }
        
        // Otherwise distribute proportionally with some randomness
        const share = Math.floor((remainingCount / (array.length - index)) * (0.7 + Math.random() * 0.6));
        remainingCount -= share;
        return { ...type, count: share };
      });
      
      return {
        city: city.city,
        country: city.state, // We'll use state as country for Indian cities
        lat: city.lat,
        lng: city.lng,
        risk: city.risk,
        scamCount,
        scamTypes
      };
    });
    
    setTimeout(() => {
      resolve(locations);
    }, 800); // Simulate API delay
  });
}

// Function to get statistics across all locations
export function getGlobalStats(locations: ScamLocation[]) {
  const totalScams = locations.reduce((sum, loc) => sum + loc.scamCount, 0);
  const avgRisk = locations.reduce((sum, loc) => sum + loc.risk, 0) / locations.length;
  
  // Get top scam types
  const typeCounts: Record<string, number> = {};
  
  locations.forEach(location => {
    location.scamTypes.forEach(type => {
      if (typeCounts[type.type]) {
        typeCounts[type.type] += type.count;
      } else {
        typeCounts[type.type] = type.count;
      }
    });
  });
  
  // Convert to array and sort
  const scamTypeBreakdown = Object.entries(typeCounts)
    .map(([type, count]) => ({
      type,
      count,
      percentage: (count / totalScams) * 100
    }))
    .sort((a, b) => b.count - a.count);
  
  // Top risk cities
  const highRiskLocations = [...locations]
    .sort((a, b) => b.risk - a.risk)
    .slice(0, 5);
  
  return {
    totalScams,
    avgRisk,
    scamTypeBreakdown,
    highRiskLocations
  };
}

// Function to add new data point
export function addScamLocation(newLocation: Omit<ScamLocation, 'scamTypes'> & { scamTypes: { type: string, count: number }[] }): Promise<ScamLocation> {
  // In a real implementation:
  // return fetch('/api/fraud-map/locations', {
  //   method: 'POST',
  //   headers: { 'Content-Type': 'application/json' },
  //   body: JSON.stringify(newLocation)
  // }).then(res => res.json());
  
  // For demo, just return the input with colors added
  const formattedLocation: ScamLocation = {
    ...newLocation,
    scamTypes: newLocation.scamTypes.map(type => {
      const matchingType = SCAM_TYPES.find(t => t.type === type.type);
      return {
        ...type,
        color: matchingType?.color || '#999999'
      };
    })
  };
  
  return new Promise(resolve => {
    setTimeout(() => {
      resolve(formattedLocation);
    }, 500);
  });
}