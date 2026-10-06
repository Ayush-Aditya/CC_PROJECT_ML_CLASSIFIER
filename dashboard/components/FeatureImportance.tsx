'use client';
import { motion } from 'framer-motion';
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer } from 'recharts';

export default function FeatureImportance({ data }: { data: any }) {
  const modelsWithFI = data.models.filter((m: any) => m.feature_importance);

  return (
    <motion.section 
      className="section"
      initial={{ opacity: 0, y: 30 }}
      whileInView={{ opacity: 1, y: 0 }}
      viewport={{ once: true, margin: '-100px' }}
      transition={{ duration: 0.6, ease: 'easeOut' }}
    >
      <h2 className="section-title">Feature Importance</h2>
      <p className="section-subtitle">Most influential predictors according to tree-based models</p>
      
      <div className="fi-grid">
        {modelsWithFI.map((model: any) => (
          <div key={model.short_name} className="fi-card">
            <h3 className="fi-card-title" style={{ color: model.color }}>{model.name}</h3>
            <div className="chart-container chart-container-sm">
              <ResponsiveContainer width="100%" height="100%">
                <BarChart 
                  data={model.feature_importance} 
                  layout="vertical"
                  margin={{ top: 5, right: 20, left: 60, bottom: 5 }}
                >
                  <CartesianGrid strokeDasharray="3 3" horizontal={false} stroke="rgba(148, 163, 184, 0.1)" />
                  <XAxis type="number" stroke="#94a3b8" />
                  <YAxis dataKey="feature" type="category" stroke="#94a3b8" width={100} tick={{ fontSize: 12 }} />
                  <Tooltip contentStyle={{ backgroundColor: 'rgba(15, 23, 42, 0.9)', borderColor: 'rgba(148, 163, 184, 0.2)', borderRadius: '8px' }} />
                  <Bar dataKey="importance" fill={model.color} radius={[0, 4, 4, 0]} />
                </BarChart>
              </ResponsiveContainer>
            </div>
          </div>
        ))}
      </div>
    </motion.section>
  );
}
