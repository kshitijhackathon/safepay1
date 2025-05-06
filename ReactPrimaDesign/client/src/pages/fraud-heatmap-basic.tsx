import React from 'react';
import MainLayout from '@/layouts/main-layout';

export default function FraudHeatmapBasic() {
  return (
    <MainLayout>
      <div className="p-6">
        <h1 className="text-2xl font-bold mb-4">Fraud Heatmap (Basic)</h1>
        <p className="mb-4">This is a simplified version of the Fraud Heatmap page.</p>
        
        <div className="bg-primary-50 p-4 rounded-xl border border-primary-200">
          <h2 className="text-lg font-bold mb-2">Map Visualization</h2>
          <p>The interactive map would be displayed here.</p>
        </div>
      </div>
    </MainLayout>
  );
}