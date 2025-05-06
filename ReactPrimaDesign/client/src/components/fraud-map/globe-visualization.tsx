import React, { useState, useEffect, useRef, useMemo } from 'react';
import Globe from 'react-globe.gl';
import * as d3 from 'd3';
import { ScamLocation } from '@/types/fraud-map';
import { Loader2, MapPin } from 'lucide-react';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Tabs, TabsList, TabsTrigger, TabsContent } from '@/components/ui/tabs';
import { PieChart, Pie, Cell, Tooltip, Legend, ResponsiveContainer } from 'recharts';
import { cn } from '@/lib/utils';

type GlobeVisualizationProps = {
  data?: ScamLocation[];
  loading?: boolean;
  className?: string;
  onLocationSelect?: (location: ScamLocation) => void;
};

export function GlobeVisualization({ 
  data = [], 
  loading = false,
  className,
  onLocationSelect
}: GlobeVisualizationProps) {
  const globeRef = useRef<any>();
  const [selectedLocation, setSelectedLocation] = useState<ScamLocation | null>(null);
  const [viewMode, setViewMode] = useState<'3d' | 'map'>('3d');
  const [windowDimensions, setWindowDimensions] = useState({
    width: typeof window !== 'undefined' ? window.innerWidth : 800,
    height: typeof window !== 'undefined' ? window.innerHeight : 600
  });
  
  // Update window dimensions on resize
  useEffect(() => {
    const handleResize = () => {
      setWindowDimensions({
        width: window.innerWidth,
        height: window.innerHeight
      });
    };
    
    window.addEventListener('resize', handleResize);
    return () => window.removeEventListener('resize', handleResize);
  }, []);
  
  // Calculate appropriate globe dimensions based on container size
  const globeDimensions = useMemo(() => {
    // For mobile
    if (windowDimensions.width < 640) {
      return {
        width: windowDimensions.width * 0.9,
        height: 400
      };
    }
    
    // For desktop
    return {
      width: Math.min(windowDimensions.width * 0.8, 900),
      height: Math.min(windowDimensions.height * 0.6, 600)
    };
  }, [windowDimensions]);
  
  // Auto-rotate globe when idle
  useEffect(() => {
    let rotationTimer: ReturnType<typeof setTimeout>;
    let lastInteraction = Date.now();
    const idleTime = 10000; // 10 seconds
    
    const checkIdle = () => {
      const now = Date.now();
      if (now - lastInteraction > idleTime && globeRef.current && viewMode === '3d') {
        globeRef.current.controls().autoRotate = true;
        globeRef.current.controls().autoRotateSpeed = 0.5;
      }
      
      rotationTimer = setTimeout(checkIdle, 1000);
    };
    
    const handleInteraction = () => {
      lastInteraction = Date.now();
      if (globeRef.current) {
        globeRef.current.controls().autoRotate = false;
      }
    };
    
    rotationTimer = setTimeout(checkIdle, idleTime);
    window.addEventListener('mousemove', handleInteraction);
    window.addEventListener('touchstart', handleInteraction);
    
    return () => {
      clearTimeout(rotationTimer);
      window.removeEventListener('mousemove', handleInteraction);
      window.removeEventListener('touchstart', handleInteraction);
    };
  }, [viewMode]);
  
  // Colorize points based on risk level
  const getPointColor = (point: ScamLocation) => {
    if (point.risk > 0.7) return 'rgba(239, 68, 68, 0.8)'; // High risk - red
    if (point.risk > 0.4) return 'rgba(245, 158, 11, 0.8)'; // Medium risk - amber
    return 'rgba(34, 197, 94, 0.8)'; // Low risk - green
  };
  
  // Calculate point size based on scam count
  const getPointSize = (point: ScamLocation) => {
    // Scale point size logarithmically based on scam count
    // Min size 0.1, max size 1.2
    return Math.max(0.1, Math.min(1.2, 0.3 + Math.log10(point.scamCount / 100) * 0.5));
  };
  
  // Format number with commas
  const formatNumber = (num: number): string => {
    return num.toString().replace(/\B(?=(\d{3})+(?!\d))/g, ",");
  };
  
  // Get color based on risk level
  const getRiskColor = (risk: number): string => {
    if (risk > 0.7) return 'text-destructive';
    if (risk > 0.4) return 'text-amber-500';
    return 'text-success';
  };
  
  // Reset view when switching modes
  useEffect(() => {
    if (globeRef.current) {
      if (viewMode === '3d') {
        globeRef.current.pointOfView({ altitude: 2.5 }, 1000);
      } else {
        globeRef.current.pointOfView({ altitude: 0.01, lat: 20, lng: 70 }, 1000);
      }
    }
  }, [viewMode]);
  
  return (
    <div className={cn("flex flex-col items-center", className)}>
      {loading ? (
        <div className="flex justify-center items-center h-[500px] w-full">
          <Loader2 className="h-12 w-12 animate-spin text-primary" />
        </div>
      ) : (
        <div className="flex flex-col items-center">
          <div className="flex justify-center gap-4 mb-4">
            <Tabs value={viewMode} onValueChange={(value) => setViewMode(value as '3d' | 'map')}>
              <TabsList>
                <TabsTrigger value="3d">3D Globe</TabsTrigger>
                <TabsTrigger value="map">Map View</TabsTrigger>
              </TabsList>
            </Tabs>
          </div>
          
          <div style={{ position: 'relative', height: globeDimensions.height, width: globeDimensions.width }}>
            <Globe
              ref={globeRef}
              globeImageUrl="//unpkg.com/three-globe/example/img/earth-blue-marble.jpg"
              bumpImageUrl="//unpkg.com/three-globe/example/img/earth-topology.png"
              backgroundColor="rgba(0,0,0,0)"
              width={globeDimensions.width}
              height={globeDimensions.height}
              pointsData={data}
              pointLat={(d: object) => (d as ScamLocation).lat}
              pointLng={(d: object) => (d as ScamLocation).lng}
              pointColor={getPointColor as any}
              pointRadius={getPointSize as any}
              pointAltitude={0.01}
              pointLabel={(d: object) => {
                const location = d as ScamLocation;
                return `
                  <div style="padding: 8px; background: rgba(0,0,0,0.8); color: white; border-radius: 4px; font-family: Arial; font-size: 12px; max-width: 200px;">
                    <b>${location.city}, ${location.country}</b><br />
                    Scam Count: ${formatNumber(location.scamCount)}<br />
                    Risk Level: ${Math.round(location.risk * 100)}%<br />
                    <i>Click for details</i>
                  </div>
                `;
              }}
              onPointClick={(point: ScamLocation) => {
                setSelectedLocation(point);
                if (onLocationSelect) onLocationSelect(point);
              }}
              onPointHover={(point: ScamLocation | null) => {
                document.body.style.cursor = point ? 'pointer' : 'default';
              }}
              atmosphereColor={viewMode === '3d' ? "lightskyblue" : "transparent"}
              atmosphereAltitude={viewMode === '3d' ? 0.1 : 0}
            />
          </div>
          
          {selectedLocation && (
            <Card className="w-full mt-4 max-w-full overflow-hidden">
              <CardHeader className="px-3 sm:px-6">
                <div className="flex items-start justify-between">
                  <div>
                    <CardTitle className="flex items-center gap-2 text-base sm:text-lg">
                      <MapPin className="h-4 w-4 flex-shrink-0" />
                      <span className="truncate">{selectedLocation.city}, {selectedLocation.country}</span>
                    </CardTitle>
                    <CardDescription className="text-xs sm:text-sm flex flex-col sm:flex-row sm:items-center gap-1 sm:gap-2">
                      <span>Total Scams: {formatNumber(selectedLocation.scamCount)}</span>
                      <span className="hidden sm:inline">|</span>
                      <span>
                        Risk Level: 
                        <span className={cn("ml-1 font-medium", getRiskColor(selectedLocation.risk))}>
                          {Math.round(selectedLocation.risk * 100)}%
                        </span>
                      </span>
                    </CardDescription>
                  </div>
                </div>
              </CardHeader>
              <CardContent className="px-2 sm:px-6">
                <div className="space-y-2 sm:space-y-4">
                  <h4 className="text-sm font-medium">Scam Type Breakdown</h4>
                  <div className="h-[250px] sm:h-[300px]">
                    <ResponsiveContainer width="100%" height="100%">
                      <PieChart>
                        <Pie
                          data={selectedLocation.scamTypes}
                          dataKey="count"
                          nameKey="type"
                          cx="50%"
                          cy="50%"
                          innerRadius={windowDimensions.width < 640 ? 40 : 60}
                          outerRadius={windowDimensions.width < 640 ? 60 : 80}
                          paddingAngle={2}
                          label={windowDimensions.width < 640 ? undefined : ({ name, percent }) => `${name}: ${(percent * 100).toFixed(1)}%`}
                          labelLine={windowDimensions.width < 640 ? false : true}
                        >
                          {selectedLocation.scamTypes.map((entry, index) => (
                            <Cell key={`cell-${index}`} fill={entry.color} />
                          ))}
                        </Pie>
                        <Tooltip 
                          formatter={(value) => [formatNumber(value as number), 'Cases']}
                        />
                        <Legend 
                          layout={windowDimensions.width < 640 ? "horizontal" : "vertical"} 
                          verticalAlign={windowDimensions.width < 640 ? "bottom" : "middle"} 
                          align={windowDimensions.width < 640 ? "center" : "right"}
                          wrapperStyle={windowDimensions.width < 640 ? { fontSize: '10px' } : undefined}
                        />
                      </PieChart>
                    </ResponsiveContainer>
                  </div>
                </div>
              </CardContent>
            </Card>
          )}
        </div>
      )}
    </div>
  );
}