import React, { useState } from 'react';
import { Button } from '@/components/ui/button';
import { 
  Select, 
  SelectContent, 
  SelectItem, 
  SelectTrigger, 
  SelectValue 
} from '@/components/ui/select';
import { Card } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { MapPin, AlertTriangle, Calendar, Search } from 'lucide-react';
import { Input } from '@/components/ui/input';
import { 
  ComposableMap, 
  Geographies, 
  Geography, 
  Marker,
  ZoomableGroup
} from 'react-simple-maps';

// Simplified India GeoJSON for the map
const INDIA_GEO_JSON = {
  "type": "FeatureCollection",
  "features": [
    {
      "type": "Feature",
      "properties": { "name": "India" },
      "geometry": {
        "type": "Polygon",
        "coordinates": [
          [
            [68.1, 7.9], [97.4, 7.9], [97.4, 35.5], [68.1, 35.5], [68.1, 7.9]
          ]
        ]
      }
    }
  ]
};

// Import UPI scam data from our CSV file structure
const fraudData = [
  { city: 'New Delhi', state: 'Delhi', cases: 450, latitude: 28.6139, longitude: 77.2090, lastReported: '2025-04-15' },
  { city: 'Mumbai', state: 'Maharashtra', cases: 620, latitude: 19.0760, longitude: 72.8777, lastReported: '2025-04-12' },
  { city: 'Bengaluru', state: 'Karnataka', cases: 300, latitude: 12.9716, longitude: 77.5946, lastReported: '2025-04-10' },
  { city: 'Kolkata', state: 'West Bengal', cases: 280, latitude: 22.5726, longitude: 88.3639, lastReported: '2025-04-11' },
  { city: 'Chennai', state: 'Tamil Nadu', cases: 400, latitude: 13.0827, longitude: 80.2707, lastReported: '2025-04-13' },
  { city: 'Hyderabad', state: 'Telangana', cases: 180, latitude: 17.3850, longitude: 78.4867, lastReported: '2025-04-17' },
  { city: 'Jaipur', state: 'Rajasthan', cases: 120, latitude: 26.9124, longitude: 75.7873, lastReported: '2025-04-14' },
  { city: 'Ahmedabad', state: 'Gujarat', cases: 220, latitude: 23.0225, longitude: 72.5714, lastReported: '2025-04-09' },
  { city: 'Pune', state: 'Maharashtra', cases: 230, latitude: 18.5204, longitude: 73.8567, lastReported: '2025-04-08' },
  { city: 'Lucknow', state: 'Uttar Pradesh', cases: 150, latitude: 26.8467, longitude: 80.9462, lastReported: '2025-04-16' },
];

// Type filter options
const timePeriods = ['Last Week', 'Last Month', 'Last 3 Months', 'Last Year', 'All Time'];

// Function to determine risk level based on scam cases
const getRiskLevel = (cases: number) => {
  if (cases > 400) return { level: 'High Risk', color: 'bg-red-500', fill: '#ef4444' };
  if (cases > 200) return { level: 'Medium Risk', color: 'bg-amber-500', fill: '#f59e0b' };
  return { level: 'Low Risk', color: 'bg-green-500', fill: '#22c55e' };
};

export default function FraudMap() {
  const [selectedPeriod, setSelectedPeriod] = useState('All Time');
  const [searchQuery, setSearchQuery] = useState('');
  const [selectedCity, setSelectedCity] = useState<string | null>(null);
  
  // Filter cities based on search query
  const filteredCities = searchQuery 
    ? fraudData.filter(city => 
        city.city.toLowerCase().includes(searchQuery.toLowerCase()) || 
        city.state.toLowerCase().includes(searchQuery.toLowerCase())
      )
    : fraudData;
  
  // Get data for selected city
  const selectedCityData = selectedCity
    ? fraudData.find(city => city.city === selectedCity)
    : null;
  
  // Sort cities by scam cases (highest first)
  const sortedCities = [...filteredCities].sort((a, b) => b.cases - a.cases);
  
  // Handle city selection
  const handleCitySelect = (city: string) => {
    setSelectedCity(city);
  };

  return (
    <div className="flex flex-col min-h-screen bg-slate-50">
      {/* Header */}
      <div className="sticky top-0 z-10 bg-white border-b p-4">
        <h1 className="text-xl font-medium">UPI Fraud Heatmap</h1>
        <p className="text-muted-foreground text-sm">
          View scam density and reports across India
        </p>
      </div>
      
      {/* Main content */}
      <div className="flex-1 p-4">
        {/* Filters */}
        <div className="mb-6">
          <Card className="p-4">
            <h2 className="text-lg font-medium mb-3">Scam Heatmap Filters</h2>
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
              <div>
                <label className="text-sm text-gray-500 mb-1 block">Time Period</label>
                <Select
                  value={selectedPeriod}
                  onValueChange={setSelectedPeriod}
                >
                  <SelectTrigger className="w-full">
                    <SelectValue placeholder="Select period" />
                  </SelectTrigger>
                  <SelectContent>
                    {timePeriods.map(period => (
                      <SelectItem key={period} value={period}>{period}</SelectItem>
                    ))}
                  </SelectContent>
                </Select>
              </div>
              <div>
                <label className="text-sm text-gray-500 mb-1 block">Search Location</label>
                <div className="relative">
                  <Search className="absolute left-2 top-2.5 h-4 w-4 text-muted-foreground" />
                  <Input
                    placeholder="Search city or state"
                    className="pl-8"
                    value={searchQuery}
                    onChange={(e) => setSearchQuery(e.target.value)}
                  />
                </div>
              </div>
            </div>
          </Card>
        </div>
        
        {/* Map placeholder */}
        <Card className="p-4 mb-6">
          <div className="flex justify-between items-center mb-4">
            <h2 className="text-lg font-medium">Scam Density Visualization</h2>
            <Badge variant="outline" className="gap-1">
              <Calendar className="h-3 w-3" />
              <span>{selectedPeriod}</span>
            </Badge>
          </div>
          
          <div className="bg-[#f5f7f9] rounded-lg p-4 flex flex-col items-center justify-center h-[350px] mb-4 border border-gray-200 overflow-hidden">
            <div className="relative w-full h-full flex items-center justify-center">
              <ComposableMap
                projectionConfig={{ scale: 600 }}
                projection="geoMercator"
                width={800}
                height={350}
                style={{ width: "100%", height: "100%" }}
              >
                <ZoomableGroup center={[80, 22]} zoom={1}>
                  <Geographies geography={INDIA_GEO_JSON}>
                    {({ geographies }) =>
                      geographies.map(geo => (
                        <Geography
                          key={geo.rsmKey}
                          geography={geo}
                          style={{
                            default: { outline: "none", fill: "#D6D6DA", stroke: "#FFFFFF" },
                            hover: { outline: "none", fill: "#F5F5F5", stroke: "#FFFFFF" },
                            pressed: { outline: "none", fill: "#D6D6DA", stroke: "#FFFFFF" }
                          }}
                        />
                      ))
                    }
                  </Geographies>
                  
                  {/* Map city markers with risk level colors */}
                  {fraudData.map((city, index) => {
                    const risk = getRiskLevel(city.cases);
                    const markerSize = city.cases > 400 ? 14 : 
                                     city.cases > 200 ? 10 : 8;
                    
                    return (
                      <Marker 
                        key={index} 
                        coordinates={[city.longitude, city.latitude]}
                        onClick={() => handleCitySelect(city.city)}
                      >
                        <circle 
                          r={markerSize} 
                          fill={risk.fill} 
                          stroke="#FFFFFF" 
                          strokeWidth={1} 
                        />
                        <text 
                          textAnchor="middle" 
                          y={-markerSize - 5} 
                          style={{ 
                            fontSize: 10, 
                            fontWeight: "bold",
                            fill: "#333333"
                          }}
                        >
                          {city.city}
                        </text>
                      </Marker>
                    );
                  })}
                </ZoomableGroup>
              </ComposableMap>
              
              <div className="absolute bottom-2 right-2 text-center z-10 bg-white/80 p-2 rounded-lg text-xs">
                <p className="font-semibold">UPI Scam Heatmap</p>
                <p className="text-muted-foreground">
                  Click on any marker for details
                </p>
              </div>
            </div>
          </div>
          
          <div className="flex justify-between text-xs text-gray-500">
            <div className="flex items-center">
              <div className="w-3 h-3 bg-green-500 rounded-full mr-1"></div>
              <span>Low Risk (0-200)</span>
            </div>
            <div className="flex items-center">
              <div className="w-3 h-3 bg-amber-500 rounded-full mr-1"></div>
              <span>Medium Risk (201-400)</span>
            </div>
            <div className="flex items-center">
              <div className="w-3 h-3 bg-red-500 rounded-full mr-1"></div>
              <span>High Risk (401+)</span>
            </div>
          </div>
        </Card>
        
        {/* City data */}
        <div className="grid grid-cols-1 md:grid-cols-3 gap-6 mb-6">
          <Card className="p-4 col-span-1 md:col-span-2">
            <h2 className="text-lg font-medium mb-3">Top Scam-Affected Cities</h2>
            <div className="space-y-4">
              {sortedCities.slice(0, 5).map((city, index) => {
                const risk = getRiskLevel(city.cases);
                return (
                  <div key={index} className="flex items-start">
                    <div className={`w-3 h-3 ${risk.color} rounded-full mt-1.5 mr-2 flex-shrink-0`}></div>
                    <div className="flex-grow">
                      <div className="flex justify-between mb-1">
                        <span className="font-medium">{city.city}, {city.state}</span>
                        <span className="text-sm text-muted-foreground">{city.cases} reports</span>
                      </div>
                      <div className="w-full bg-gray-100 h-2 rounded-full overflow-hidden">
                        <div
                          className={`h-full ${risk.color}`}
                          style={{ width: `${Math.min(100, (city.cases / 620) * 100)}%` }}
                        ></div>
                      </div>
                      <div className="flex justify-between mt-1">
                        <span className="text-xs text-muted-foreground">Last reported: {city.lastReported}</span>
                        <Button 
                          variant="ghost" 
                          size="sm" 
                          className="text-xs h-6 px-2"
                          onClick={() => handleCitySelect(city.city)}
                        >
                          View details
                        </Button>
                      </div>
                    </div>
                  </div>
                );
              })}
            </div>
          </Card>
          
          <Card className="p-4">
            <h2 className="text-lg font-medium mb-3">Risk Summary</h2>
            <div className="space-y-4">
              <div>
                <div className="flex items-center justify-between mb-2">
                  <span className="text-sm font-medium">High Risk Cities</span>
                  <Badge variant="destructive">{fraudData.filter(city => city.cases > 400).length}</Badge>
                </div>
                <div className="flex items-center justify-between mb-2">
                  <span className="text-sm font-medium">Medium Risk Cities</span>
                  <Badge variant="outline" className="bg-amber-100">{fraudData.filter(city => city.cases > 200 && city.cases <= 400).length}</Badge>
                </div>
                <div className="flex items-center justify-between">
                  <span className="text-sm font-medium">Low Risk Cities</span>
                  <Badge variant="outline" className="bg-green-100">{fraudData.filter(city => city.cases <= 200).length}</Badge>
                </div>
              </div>
              
              <div className="border-t pt-4 mt-4">
                <div className="text-sm font-medium mb-2">Total Reports</div>
                <div className="text-3xl font-bold">{fraudData.reduce((sum, city) => sum + city.cases, 0)}</div>
                <div className="text-sm text-muted-foreground">Across all cities</div>
              </div>
              
              <div className="border-t pt-4 mt-4">
                <div className="text-sm font-medium mb-2">Highest Risk City</div>
                <div className="text-xl font-semibold">{sortedCities[0].city}</div>
                <div className="text-sm text-muted-foreground">{sortedCities[0].cases} reports</div>
              </div>
            </div>
          </Card>
        </div>
        
        {/* Selected city details */}
        {selectedCity && selectedCityData && (
          <Card className="p-4 mb-6">
            <div className="flex justify-between items-center mb-4">
              <h2 className="text-lg font-medium">{selectedCityData.city} Details</h2>
              <Button variant="ghost" size="sm" onClick={() => setSelectedCity(null)}>
                <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                  <line x1="18" y1="6" x2="6" y2="18"></line>
                  <line x1="6" y1="6" x2="18" y2="18"></line>
                </svg>
              </Button>
            </div>
            
            <div className="grid grid-cols-2 md:grid-cols-4 gap-4 mb-4">
              <div className="bg-slate-50 p-3 rounded-lg">
                <div className="text-sm text-muted-foreground">Total Reports</div>
                <div className="text-2xl font-semibold">{selectedCityData.cases}</div>
              </div>
              <div className="bg-slate-50 p-3 rounded-lg">
                <div className="text-sm text-muted-foreground">Risk Level</div>
                <div className="text-2xl font-semibold flex items-center gap-2">
                  <span className={`w-2 h-2 ${getRiskLevel(selectedCityData.cases).color} rounded-full`}></span>
                  {getRiskLevel(selectedCityData.cases).level.split(' ')[0]}
                </div>
              </div>
              <div className="bg-slate-50 p-3 rounded-lg">
                <div className="text-sm text-muted-foreground">State</div>
                <div className="text-2xl font-semibold">{selectedCityData.state}</div>
              </div>
              <div className="bg-slate-50 p-3 rounded-lg">
                <div className="text-sm text-muted-foreground">Last Reported</div>
                <div className="text-2xl font-semibold">{selectedCityData.lastReported}</div>
              </div>
            </div>
            
            <div className="bg-slate-50 p-4 rounded-lg mb-4">
              <div className="flex items-center gap-2 mb-2">
                <AlertTriangle className="h-4 w-4 text-amber-500" />
                <h3 className="font-medium">Common Scam Types</h3>
              </div>
              <div className="grid grid-cols-2 gap-2">
                <div className="flex justify-between items-center">
                  <span className="text-sm">Banking Scams</span>
                  <span className="text-sm font-medium">{Math.floor(selectedCityData.cases * 0.4)}</span>
                </div>
                <div className="flex justify-between items-center">
                  <span className="text-sm">UPI Phishing</span>
                  <span className="text-sm font-medium">{Math.floor(selectedCityData.cases * 0.3)}</span>
                </div>
                <div className="flex justify-between items-center">
                  <span className="text-sm">KYC Fraud</span>
                  <span className="text-sm font-medium">{Math.floor(selectedCityData.cases * 0.2)}</span>
                </div>
                <div className="flex justify-between items-center">
                  <span className="text-sm">Others</span>
                  <span className="text-sm font-medium">{Math.floor(selectedCityData.cases * 0.1)}</span>
                </div>
              </div>
            </div>
            
            <div className="flex justify-center">
              <Button 
                className="gap-2"
              >
                <MapPin className="h-4 w-4" />
                <span>View on Map</span>
              </Button>
            </div>
          </Card>
        )}
        
        {/* Data notice */}
        <p className="text-xs text-muted-foreground">
          Data last updated: April 17, 2025. The fraud heatmap is based on actual UPI scam reports and fraud alerts from our database.
        </p>
      </div>
    </div>
  );
}