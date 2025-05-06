import React, { useState, useEffect } from 'react';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { GlobeVisualization } from '@/components/fraud-map/globe-visualization';
import { getScamLocationData, getGlobalStats } from '@/services/fraud-map-service';
import { ScamLocation, GlobalStats } from '@/types/fraud-map';
import { Loader2, AlertCircle, MapPin, Info } from 'lucide-react';
import { Alert, AlertTitle, AlertDescription } from '@/components/ui/alert';
import { PageLayout } from '@/layouts/page-layout';
import { PieChart, Pie, Cell, Tooltip, Legend, ResponsiveContainer, BarChart, Bar, XAxis, YAxis, CartesianGrid } from 'recharts';
import { cn } from '@/lib/utils';

export default function FraudMapPage() {
  const [locations, setLocations] = useState<ScamLocation[]>([]);
  const [stats, setStats] = useState<GlobalStats | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [selectedLocation, setSelectedLocation] = useState<ScamLocation | null>(null);
  const [viewMode, setViewMode] = useState<'globe' | 'stats'>('globe');
  
  // Fetch location data
  useEffect(() => {
    const fetchData = async () => {
      try {
        setLoading(true);
        const data = await getScamLocationData();
        setLocations(data);
        
        // Calculate global stats
        const globalStats = getGlobalStats(data);
        setStats(globalStats);
        
        setError(null);
      } catch (err) {
        console.error('Error fetching fraud map data:', err);
        setError('Failed to load fraud map data. Please try again later.');
      } finally {
        setLoading(false);
      }
    };
    
    fetchData();
  }, []);
  
  // Handle location selection
  const handleLocationSelect = (location: ScamLocation) => {
    setSelectedLocation(location);
  };
  
  // Get color based on risk level
  const getRiskColor = (risk: number): string => {
    if (risk > 0.7) return 'text-destructive';
    if (risk > 0.4) return 'text-amber-500';
    return 'text-success';
  };
  
  // Format number with commas
  const formatNumber = (num: number): string => {
    return num.toString().replace(/\B(?=(\d{3})+(?!\d))/g, ",");
  };
  
  return (
    <PageLayout>
      <div className="space-y-4">
        <div className="flex flex-col sm:flex-row justify-between items-start sm:items-center gap-4">
          <div>
            <h1 className="text-2xl font-bold tracking-tight">Global Scam Risk Map</h1>
            <p className="text-muted-foreground">
              Interactive visualization of scam activity and fraud patterns worldwide
            </p>
          </div>
          
          <Tabs
            value={viewMode}
            onValueChange={(value) => setViewMode(value as 'globe' | 'stats')}
            className="w-full sm:w-auto"
          >
            <TabsList>
              <TabsTrigger value="globe">3D Globe</TabsTrigger>
              <TabsTrigger value="stats">Statistics</TabsTrigger>
            </TabsList>
          </Tabs>
        </div>
        
        {error && (
          <Alert variant="destructive">
            <AlertCircle className="h-4 w-4" />
            <AlertTitle>Error</AlertTitle>
            <AlertDescription>{error}</AlertDescription>
          </Alert>
        )}
        
        <Tabs value={viewMode} className="hidden">
          <TabsContent value="globe" className="m-0">
            {/* 3D Globe Visualization */}
            <Card className="border-0 shadow-none bg-transparent">
              <CardContent className="p-0">
                <GlobeVisualization 
                  data={locations} 
                  loading={loading}
                  onLocationSelect={handleLocationSelect}
                />
              </CardContent>
            </Card>
          </TabsContent>
          
          <TabsContent value="stats" className="m-0">
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
              {/* Global Statistics Card */}
              <Card>
                <CardHeader>
                  <CardTitle>Global Statistics</CardTitle>
                  <CardDescription>
                    Overview of fraud activity across all monitored locations
                  </CardDescription>
                </CardHeader>
                <CardContent>
                  {loading ? (
                    <div className="flex justify-center items-center h-[300px]">
                      <Loader2 className="h-8 w-8 animate-spin text-primary" />
                    </div>
                  ) : stats ? (
                    <div className="space-y-6">
                      <div className="grid grid-cols-2 gap-4">
                        <div className="bg-muted/50 p-4 rounded-lg">
                          <div className="text-3xl font-bold">
                            {formatNumber(stats.totalScams)}
                          </div>
                          <div className="text-sm text-muted-foreground">
                            Total Scams Detected
                          </div>
                        </div>
                        <div className="bg-muted/50 p-4 rounded-lg">
                          <div className={cn(
                            "text-3xl font-bold",
                            getRiskColor(stats.avgRisk)
                          )}>
                            {(stats.avgRisk * 100).toFixed(1)}%
                          </div>
                          <div className="text-sm text-muted-foreground">
                            Average Risk Level
                          </div>
                        </div>
                      </div>
                      
                      <div>
                        <h4 className="text-sm font-medium mb-2">Scam Type Distribution</h4>
                        <div className="h-[250px]">
                          <ResponsiveContainer width="100%" height="100%">
                            <PieChart>
                              <Pie
                                data={stats.scamTypeBreakdown.slice(0, 5)}
                                dataKey="count"
                                nameKey="type"
                                cx="50%"
                                cy="50%"
                                innerRadius={60}
                                outerRadius={80}
                                paddingAngle={2}
                                label={({ name, percent }) => `${name}: ${(percent * 100).toFixed(1)}%`}
                                labelLine={false}
                              >
                                {stats.scamTypeBreakdown.slice(0, 5).map((entry, index) => (
                                  <Cell key={`cell-${index}`} fill={`hsl(${index * 36}, 70%, 50%)`} />
                                ))}
                              </Pie>
                              <Tooltip 
                                formatter={(value) => [formatNumber(value as number), 'Cases']}
                              />
                              <Legend layout="vertical" verticalAlign="middle" align="right" />
                            </PieChart>
                          </ResponsiveContainer>
                        </div>
                      </div>
                    </div>
                  ) : null}
                </CardContent>
              </Card>
              
              {/* High Risk Locations Card */}
              <Card>
                <CardHeader>
                  <CardTitle>Highest Risk Locations</CardTitle>
                  <CardDescription>
                    Cities with the highest scam risk levels
                  </CardDescription>
                </CardHeader>
                <CardContent>
                  {loading ? (
                    <div className="flex justify-center items-center h-[300px]">
                      <Loader2 className="h-8 w-8 animate-spin text-primary" />
                    </div>
                  ) : stats ? (
                    <div className="space-y-6">
                      <div className="h-[300px]">
                        <ResponsiveContainer width="100%" height="100%">
                          <BarChart
                            data={stats.highRiskLocations.map(loc => ({
                              name: loc.city,
                              risk: Math.round(loc.risk * 100),
                              scams: loc.scamCount
                            }))}
                            layout="vertical"
                            margin={{ top: 5, right: 30, left: 20, bottom: 5 }}
                          >
                            <CartesianGrid strokeDasharray="3 3" />
                            <XAxis type="number" domain={[0, 100]} label={{ value: 'Risk %', position: 'bottom' }} />
                            <YAxis type="category" dataKey="name" width={100} />
                            <Tooltip 
                              formatter={(value, name) => [
                                name === 'risk' ? `${value}%` : formatNumber(value as number), 
                                name === 'risk' ? 'Risk Level' : 'Total Scams'
                              ]} 
                            />
                            <Bar dataKey="risk" fill="#ef4444" name="Risk Level" />
                          </BarChart>
                        </ResponsiveContainer>
                      </div>
                      
                      <Alert>
                        <Info className="h-4 w-4" />
                        <AlertTitle>Risk Calculation</AlertTitle>
                        <AlertDescription>
                          Risk levels are calculated based on scam frequency, severity, and population factors.
                        </AlertDescription>
                      </Alert>
                    </div>
                  ) : null}
                </CardContent>
              </Card>
            </div>
          </TabsContent>
        </Tabs>
        
        {/* Always show the active view */}
        {viewMode === 'globe' ? (
          <Card className="border-0 shadow-none bg-transparent">
            <CardContent className="p-0">
              <GlobeVisualization 
                data={locations} 
                loading={loading}
                onLocationSelect={handleLocationSelect}
              />
            </CardContent>
          </Card>
        ) : (
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            {/* Global Statistics Card */}
            <Card>
              <CardHeader>
                <CardTitle>Global Statistics</CardTitle>
                <CardDescription>
                  Overview of fraud activity across all monitored locations
                </CardDescription>
              </CardHeader>
              <CardContent>
                {loading ? (
                  <div className="flex justify-center items-center h-[300px]">
                    <Loader2 className="h-8 w-8 animate-spin text-primary" />
                  </div>
                ) : stats ? (
                  <div className="space-y-6">
                    <div className="grid grid-cols-2 gap-4">
                      <div className="bg-muted/50 p-4 rounded-lg">
                        <div className="text-3xl font-bold">
                          {formatNumber(stats.totalScams)}
                        </div>
                        <div className="text-sm text-muted-foreground">
                          Total Scams Detected
                        </div>
                      </div>
                      <div className="bg-muted/50 p-4 rounded-lg">
                        <div className={cn(
                          "text-3xl font-bold",
                          getRiskColor(stats.avgRisk)
                        )}>
                          {(stats.avgRisk * 100).toFixed(1)}%
                        </div>
                        <div className="text-sm text-muted-foreground">
                          Average Risk Level
                        </div>
                      </div>
                    </div>
                    
                    <div>
                      <h4 className="text-sm font-medium mb-2">Scam Type Distribution</h4>
                      <div className="h-[250px]">
                        <ResponsiveContainer width="100%" height="100%">
                          <PieChart>
                            <Pie
                              data={stats.scamTypeBreakdown.slice(0, 5)}
                              dataKey="count"
                              nameKey="type"
                              cx="50%"
                              cy="50%"
                              innerRadius={60}
                              outerRadius={80}
                              paddingAngle={2}
                              label={({ name, percent }) => `${name}: ${(percent * 100).toFixed(1)}%`}
                              labelLine={false}
                            >
                              {stats.scamTypeBreakdown.slice(0, 5).map((entry, index) => (
                                <Cell key={`cell-${index}`} fill={`hsl(${index * 36}, 70%, 50%)`} />
                              ))}
                            </Pie>
                            <Tooltip 
                              formatter={(value) => [formatNumber(value as number), 'Cases']}
                            />
                            <Legend layout="vertical" verticalAlign="middle" align="right" />
                          </PieChart>
                        </ResponsiveContainer>
                      </div>
                    </div>
                  </div>
                ) : null}
              </CardContent>
            </Card>
            
            {/* High Risk Locations Card */}
            <Card>
              <CardHeader>
                <CardTitle>Highest Risk Locations</CardTitle>
                <CardDescription>
                  Cities with the highest scam risk levels
                </CardDescription>
              </CardHeader>
              <CardContent>
                {loading ? (
                  <div className="flex justify-center items-center h-[300px]">
                    <Loader2 className="h-8 w-8 animate-spin text-primary" />
                  </div>
                ) : stats ? (
                  <div className="space-y-6">
                    <div className="h-[300px]">
                      <ResponsiveContainer width="100%" height="100%">
                        <BarChart
                          data={stats.highRiskLocations.map(loc => ({
                            name: loc.city,
                            risk: Math.round(loc.risk * 100),
                            scams: loc.scamCount
                          }))}
                          layout="vertical"
                          margin={{ top: 5, right: 30, left: 20, bottom: 5 }}
                        >
                          <CartesianGrid strokeDasharray="3 3" />
                          <XAxis type="number" domain={[0, 100]} label={{ value: 'Risk %', position: 'bottom' }} />
                          <YAxis type="category" dataKey="name" width={100} />
                          <Tooltip 
                            formatter={(value, name) => [
                              name === 'risk' ? `${value}%` : formatNumber(value as number), 
                              name === 'risk' ? 'Risk Level' : 'Total Scams'
                            ]} 
                          />
                          <Bar dataKey="risk" fill="#ef4444" name="Risk Level" />
                        </BarChart>
                      </ResponsiveContainer>
                    </div>
                    
                    <Alert>
                      <Info className="h-4 w-4" />
                      <AlertTitle>Risk Calculation</AlertTitle>
                      <AlertDescription>
                        Risk levels are calculated based on scam frequency, severity, and population factors.
                      </AlertDescription>
                    </Alert>
                  </div>
                ) : null}
              </CardContent>
            </Card>
          </div>
        )}
        
        {/* User instructions */}
        <Card className="bg-muted/30">
          <CardContent className="pt-6">
            <div className="flex items-start gap-4">
              <MapPin className="h-10 w-10 text-primary flex-shrink-0" />
              <div>
                <h3 className="font-medium mb-1">How to use this visualization</h3>
                <ul className="text-sm text-muted-foreground space-y-1">
                  <li>• Click and drag to rotate the globe</li>
                  <li>• Scroll to zoom in and out</li>
                  <li>• Hover over points to see basic location information</li>
                  <li>• Click on a location to view detailed scam type breakdown</li>
                  <li>• Use the map view toggle for a flat projection</li>
                  <li>• Switch to Statistics tab for global overview</li>
                </ul>
              </div>
            </div>
          </CardContent>
        </Card>
      </div>
    </PageLayout>
  );
}