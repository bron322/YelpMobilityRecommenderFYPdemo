import React from 'react';
import {
  Users,
  Star,
  Store,
  MessageSquare,
  MapPin,
  Info,
  Map,
} from 'lucide-react';
import { DatasetStats, CityData } from '@/data/dashboardData';
import { Button } from '@/components/ui/button';
import ChatbotWidget from '@/components/ChatbotWidget';

interface OverviewTabProps {
  stats: DatasetStats;
  topCities: CityData[];
  topStates: { state: string; fullName: string; businessCount: number }[];
}

const StatCard: React.FC<{
  icon: React.ReactNode;
  label: string;
  value: string | number;
  subtitle: string;
  color: 'purple' | 'orange' | 'teal' | 'coral' | 'mint';
}> = ({ icon, label, value, subtitle, color }) => (
  <div className={`stat-card ${color} animate-fade-in`}>
    <div className="flex items-start justify-between mb-3">
      <div className={`icon-container ${color}`}>{icon}</div>
    </div>
    <p className="text-3xl font-bold text-foreground mb-1">
      {typeof value === 'number' ? value.toLocaleString() : value}
    </p>
    <p className="text-sm font-semibold text-foreground mb-0.5">{label}</p>
    <p className="text-xs text-muted-foreground">{subtitle}</p>
  </div>
);

const OverviewTab: React.FC<OverviewTabProps> = ({
  stats,
  topCities,
  topStates,
}) => {
  return (
    <div className="grid grid-cols-1 lg:grid-cols-2 gap-6 animate-fade-in">
      {/* Left Column: Dataset Snapshot */}
      <div className="space-y-6">
        <div>
          <h2 className="section-title flex items-center gap-2">
            <span className="w-2 h-2 rounded-full bg-purple" />
            Dataset Snapshot
          </h2>
          <p className="section-subtitle">
            Key stats from your sampled user data
          </p>
        </div>

        <div className="grid grid-cols-2 md:grid-cols-3 gap-4">
          <StatCard
            icon={<Users className="w-5 h-5" />}
            label="Total Users"
            value={stats.totalUsers}
            subtitle="Randomly sampled"
            color="purple"
          />
          <StatCard
            icon={<Star className="w-5 h-5" />}
            label="Reviews"
            value={stats.totalReviews}
            subtitle="User ratings & feedback"
            color="orange"
          />
          <StatCard
            icon={<Store className="w-5 h-5" />}
            label="Restaurants"
            value={stats.totalBusinesses}
            subtitle="Unique places visited"
            color="teal"
          />
          <StatCard
            icon={<MessageSquare className="w-5 h-5" />}
            label="Tips"
            value={stats.totalTips}
            subtitle="Quick user insights"
            color="coral"
          />
          <StatCard
            icon={<MapPin className="w-5 h-5" />}
            label="Check-ins"
            value={stats.totalCheckins}
            subtitle="Location visits logged"
            color="mint"
          />
        </div>

        {/* Explanation callout */}
        <div className="friendly-callout">
          <div className="flex items-start gap-3">
            <div className="icon-container purple shrink-0">
              <Info className="w-5 h-5" />
            </div>
            <div>
              <h3 className="font-bold text-foreground mb-1">
                Why 10,000 users?
              </h3>
              <p className="text-sm text-muted-foreground leading-relaxed">
                We randomly sampled 10,000 users so the analysis represents
                typical user behavior, not just the first entries in the dataset
                file. This gives us a balanced view of how different people
                explore restaurants!
              </p>
            </div>
          </div>
        </div>
      </div>

      {/* Right Column: Location Distribution */}
      <div className="space-y-6">
        <div className="flex items-center justify-between">
          <div>
            <h2 className="section-title flex items-center gap-2">
              <span className="w-2 h-2 rounded-full bg-teal" />
              Where are restaurants located?
            </h2>
            <p className="section-subtitle">
              Geographic distribution of businesses
            </p>
          </div>
          <Button variant="outline" size="sm" className="gap-2 rounded-full">
            <Map className="w-4 h-4" />
            View on map
          </Button>
        </div>

        {/* Top Cities Table */}
        <div className="dashboard-card">
          <h3 className="font-bold text-foreground mb-4 flex items-center gap-2">
            <span className="chip-orange">Cities</span>
            Top locations by restaurant count
          </h3>
          <table className="data-table">
            <thead>
              <tr>
                <th>City</th>
                <th>State</th>
                <th className="text-right">Restaurants</th>
                <th className="text-right">Reviews</th>
              </tr>
            </thead>
            <tbody>
              {topCities.map((city, index) => (
                <tr key={city.city} className="animate-fade-in" style={{ animationDelay: `${index * 50}ms` }}>
                  <td className="font-medium">{city.city}</td>
                  <td className="text-muted-foreground">{city.state}</td>
                  <td className="text-right font-semibold">
                    {city.businessCount.toLocaleString()}
                  </td>
                  <td className="text-right text-muted-foreground">
                    {city.sampleReviews.toLocaleString()}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>

        {/* Top States Table */}
        <div className="dashboard-card">
          <h3 className="font-bold text-foreground mb-4 flex items-center gap-2">
            <span className="chip-teal">States</span>
            Coverage by state
          </h3>
          <table className="data-table">
            <thead>
              <tr>
                <th>State</th>
                <th>Full Name</th>
                <th className="text-right">Restaurants</th>
              </tr>
            </thead>
            <tbody>
              {topStates.map((state, index) => (
                <tr key={state.state} className="animate-fade-in" style={{ animationDelay: `${index * 50}ms` }}>
                  <td>
                    <span className="inline-flex items-center justify-center w-8 h-8 rounded-lg bg-secondary font-bold text-sm">
                      {state.state}
                    </span>
                  </td>
                  <td className="font-medium">{state.fullName}</td>
                  <td className="text-right font-semibold">
                    {state.businessCount.toLocaleString()}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>

      {/* Floating Chatbot Widget */}
      <ChatbotWidget />
    </div>
    
  );
};

export default OverviewTab;
