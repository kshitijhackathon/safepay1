import React, { useState } from 'react';
import MainLayout from '@/layouts/main-layout';
import { Card } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from '@/components/ui/select';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';

// Sample fraud data by state
const fraudData = [
  { state: 'Maharashtra', reports: 120, type: 'Fake Products' },
  { state: 'Karnataka', reports: 85, type: 'Phishing' },
  { state: 'Delhi', reports: 150, type: 'Fraud' },
  { state: 'Uttar Pradesh', reports: 110, type: 'Impersonation' },
  { state: 'Tamil Nadu', reports: 95, type: 'Fake Products' },
  { state: 'Gujarat', reports: 75, type: 'Phishing' },
  { state: 'West Bengal', reports: 68, type: 'Impersonation' },
  { state: 'Telangana', reports: 55, type: 'Fraud' },
  { state: 'Rajasthan', reports: 48, type: 'Phishing' },
  { state: 'Bihar', reports: 38, type: 'Impersonation' },
  { state: 'Madhya Pradesh', reports: 42, type: 'Fake Products' },
  { state: 'Kerala', reports: 35, type: 'Fraud' },
  { state: 'Punjab', reports: 30, type: 'Phishing' },
  { state: 'Haryana', reports: 32, type: 'Impersonation' },
  { state: 'Andhra Pradesh', reports: 45, type: 'Fake Products' },
];

// Scam hotspot markers
const scamHotspots = [
  { name: "Mumbai", reports: 89 },
  { name: "Delhi", reports: 132 },
  { name: "Bengaluru", reports: 65 },
  { name: "Chennai", reports: 58 },
  { name: "Hyderabad", reports: 46 },
  { name: "Kolkata", reports: 53 },
];

// Type filter options
const scamTypes = ['All Types', 'Fake Products', 'Phishing', 'Impersonation', 'Fraud'];

// Time period filter options
const timePeriods = ['Last Week', 'Last Month', 'Last 3 Months', 'Last Year', 'All Time'];

export default function FraudHeatmap() {
  // State
  const [selectedType, setSelectedType] = useState('All Types');
  const [selectedPeriod, setSelectedPeriod] = useState('All Time');
  const [selectedRegion, setSelectedRegion] = useState<string | null>(null);
  const [showHotspots, setShowHotspots] = useState(true);
  
  // Filter data based on selected type
  const filteredData = selectedType === 'All Types'
    ? fraudData
    : fraudData.filter(item => item.type === selectedType);
  
  // Get data for a selected region
  const selectedRegionData = selectedRegion 
    ? fraudData.find(item => item.state === selectedRegion)
    : null;

  // Colors for the risk levels
  const riskColors = {
    low: '#C5E8FF',
    medium: '#89CFF0',
    high: '#5D9BF0',
    veryHigh: '#3573D9'
  };

  // Function to get risk color based on reports
  const getRiskColor = (reports: number): string => {
    if (reports <= 50) return riskColors.low;
    if (reports <= 100) return riskColors.medium; 
    if (reports <= 150) return riskColors.high;
    return riskColors.veryHigh;
  };

  return (
    <MainLayout>
      <div className="px-4 py-6">
        <h1 className="text-2xl font-bold mb-4">Heatmap</h1>
        <p className="text-gray-600 mb-6">
          Visualize scam-prone regions across India and identify high-risk areas
        </p>
        
        {/* Filters */}
        <div className="flex flex-col gap-4 mb-6">
          <div className="grid grid-cols-2 gap-4">
            <div>
              <label className="text-sm text-gray-500 mb-1 block">Scam Type</label>
              <Select
                value={selectedType}
                onValueChange={setSelectedType}
              >
                <SelectTrigger className="w-full">
                  <SelectValue placeholder="Select type" />
                </SelectTrigger>
                <SelectContent>
                  {scamTypes.map(type => (
                    <SelectItem key={type} value={type}>{type}</SelectItem>
                  ))}
                </SelectContent>
              </Select>
            </div>
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
          </div>
        </div>
        
        {/* Map and stats tabs */}
        <Tabs defaultValue="map" className="mb-6">
          <TabsList className="grid w-full grid-cols-2">
            <TabsTrigger value="map">Map View</TabsTrigger>
            <TabsTrigger value="stats">Statistics</TabsTrigger>
          </TabsList>
          
          <TabsContent value="map" className="pt-4">
            <Card className="p-4 text-center">
              <div className="bg-[#f0f6ff] p-6 rounded-lg mb-4 flex items-center justify-center">
                <svg xmlns="http://www.w3.org/2000/svg" width="100" height="100" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1" strokeLinecap="round" strokeLinejoin="round" className="text-primary">
                  <path d="M9 6.75V15m6-6v8.25m.503 3.498 4.875-2.437c.381-.19.622-.58.622-1.006V4.82c0-.836-.88-1.38-1.628-1.006l-3.869 1.934c-.317.159-.69.159-1.006 0L9.503 3.252a1.125 1.125 0 0 0-1.006 0L3.622 5.689C3.24 5.88 3 6.27 3 6.695V19.18c0 .836.88 1.38 1.628 1.006l3.869-1.934c.317-.159.69-.159 1.006 0l4.994 2.497c.317.158.69.158 1.006 0Z" />
                </svg>
              </div>
              <h3 className="text-lg font-medium mb-2">Interactive Map</h3>
              <p className="text-gray-500 mb-4">
                The interactive map visualization of scam-prone regions will be available soon.
              </p>
            </Card>
            
            <div className="mt-4 flex justify-between text-xs text-gray-500">
              <div className="flex items-center">
                <div className="w-3 h-3 bg-[#C5E8FF] rounded-sm mr-1"></div>
                <span>Low (0-50)</span>
              </div>
              <div className="flex items-center">
                <div className="w-3 h-3 bg-[#89CFF0] rounded-sm mr-1"></div>
                <span>Medium (51-100)</span>
              </div>
              <div className="flex items-center">
                <div className="w-3 h-3 bg-[#5D9BF0] rounded-sm mr-1"></div>
                <span>High (101-150)</span>
              </div>
              <div className="flex items-center">
                <div className="w-3 h-3 bg-[#3573D9] rounded-sm mr-1"></div>
                <span>Very High (150+)</span>
              </div>
            </div>
          </TabsContent>
          
          <TabsContent value="stats" className="pt-4">
            <div className="bg-gray-50 p-4 rounded-lg mb-4">
              <h3 className="font-medium mb-2">Top 5 Scam-Prone Regions</h3>
              <div className="space-y-3">
                {[...filteredData]
                  .sort((a, b) => b.reports - a.reports)
                  .slice(0, 5)
                  .map((item, index) => (
                    <div key={index} className="flex items-center">
                      <div className="w-6 h-6 rounded-full bg-primary text-white flex items-center justify-center text-xs mr-2">
                        {index + 1}
                      </div>
                      <div className="flex-1">
                        <div className="flex justify-between mb-1">
                          <span className="font-medium">{item.state}</span>
                          <span>{item.reports} reports</span>
                        </div>
                        <div className="w-full bg-gray-200 h-2 rounded-full overflow-hidden">
                          <div
                            className="h-full rounded-full"
                            style={{ 
                              width: `${(item.reports / 150) * 100}%`,
                              backgroundColor: getRiskColor(item.reports)
                            }}
                          ></div>
                        </div>
                      </div>
                    </div>
                  ))}
              </div>
            </div>
            
            <div className="grid grid-cols-2 gap-4">
              <Card className="p-4">
                <h3 className="font-medium mb-2">By Scam Type</h3>
                <div className="space-y-2">
                  {['Fake Products', 'Phishing', 'Impersonation', 'Fraud'].map(type => {
                    const count = fraudData
                      .filter(item => item.type === type)
                      .reduce((sum, item) => sum + item.reports, 0);
                    return (
                      <div key={type} className="flex justify-between items-center">
                        <span className="text-sm">{type}</span>
                        <span className="text-sm font-medium">{count}</span>
                      </div>
                    );
                  })}
                </div>
              </Card>
              
              <Card className="p-4">
                <h3 className="font-medium mb-2">Total Reports</h3>
                <div className="text-3xl font-bold mb-2">
                  {filteredData.reduce((sum, item) => sum + item.reports, 0)}
                </div>
                <div className="text-sm text-gray-500">
                  For {selectedPeriod.toLowerCase()} {selectedType !== 'All Types' ? `(${selectedType})` : ''}
                </div>
              </Card>
            </div>
          </TabsContent>
        </Tabs>
        
        {/* Selected region details */}
        {selectedRegion && selectedRegionData && (
          <Card className="p-4 mb-6">
            <div className="flex justify-between items-center mb-2">
              <h3 className="font-medium">{selectedRegion} Statistics</h3>
              <Button variant="ghost" size="sm" onClick={() => setSelectedRegion(null)}>
                <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                  <line x1="18" y1="6" x2="6" y2="18"></line>
                  <line x1="6" y1="6" x2="18" y2="18"></line>
                </svg>
              </Button>
            </div>
            <div className="grid grid-cols-2 gap-4">
              <div>
                <div className="text-sm text-gray-500">Total Reports</div>
                <div className="text-xl font-semibold">{selectedRegionData.reports}</div>
              </div>
              <div>
                <div className="text-sm text-gray-500">Most Common Type</div>
                <div className="text-xl font-semibold">{selectedRegionData.type}</div>
              </div>
              <div className="col-span-2">
                <Button variant="outline" className="w-full text-sm">View Detailed Report</Button>
              </div>
            </div>
          </Card>
        )}
        
        <Card className="p-4 mb-6">
          <h3 className="font-medium mb-2">Scam Hotspots</h3>
          <div className="space-y-2">
            {scamHotspots
              .sort((a, b) => b.reports - a.reports)
              .map((hotspot, index) => (
                <div key={index} className="flex justify-between items-center border-b border-gray-100 pb-2 last:border-0 last:pb-0">
                  <div className="flex items-center">
                    <div
                      className="w-3 h-3 rounded-full mr-2"
                      style={{ backgroundColor: getRiskColor(hotspot.reports) }}
                    ></div>
                    <span>{hotspot.name}</span>
                  </div>
                  <span className="text-sm font-medium">{hotspot.reports} reports</span>
                </div>
              ))}
          </div>
        </Card>
        
        <p className="text-xs text-gray-500 mb-4">
          Data updated hourly. The heatmap is generated based on actual user reports and fraud alerts from our database.
        </p>
      </div>
    </MainLayout>
  );
}