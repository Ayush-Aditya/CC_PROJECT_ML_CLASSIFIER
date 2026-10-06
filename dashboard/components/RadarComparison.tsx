'use client';
import { motion } from 'framer-motion';
import { Radar, RadarChart, PolarGrid, PolarAngleAxis, PolarRadiusAxis, ResponsiveContainer, Legend, Tooltip } from 'recharts';

export default function RadarComparison({ data }: { data: any }) {
  const metrics = ['accuracy', 'precision', 'recall', 'f1_score', 'auc'];
  
  const chartData = metrics.map(metric => {
    const obj: any = { metric: metric.charAt(0).toUpperCase() + metric.slice(1).replace('_', ' ') };
    data.models.forEach((m: any) => {
      obj[m.short_name] = m.metrics[metric];
    });
    return obj;
  });

  return (
    <motion.section 
      className="section"
      initial={{ opacity: 0, y: 30 }}
      whileInView={{ opacity: 1, y: 0 }}
      viewport={{ once: true, margin: '-100px' }}
      transition={{ duration: 0.6, ease: 'easeOut' }}
    >
      <h2 className="section-title">Multi-Dimensional Comparison</h2>
      <p className="section-subtitle">Radar chart showing model performance across all metrics</p>
      
      <div className="chart-container chart-container-lg">
        <ResponsiveContainer width="100%" height="100%">
          <RadarChart cx="50%" cy="50%" outerRadius="70%" data={chartData}>
            <PolarGrid stroke="rgba(148, 163, 184, 0.2)" />
            <PolarAngleAxis dataKey="metric" tick={{ fill: '#94a3b8', fontSize: 14 }} />
            <PolarRadiusAxis angle={30} domain={[0.5, 1]} tick={{ fill: '#64748b' }} />
            <Tooltip contentStyle={{ backgroundColor: 'rgba(15, 23, 42, 0.9)', borderColor: 'rgba(148, 163, 184, 0.2)', borderRadius: '8px' }} />
            <Legend wrapperStyle={{ paddingTop: '20px' }} />
            {data.models.map((model: any) => (
              <Radar 
                key={model.short_name}
                name={model.name}
                dataKey={model.short_name}
                stroke={model.color}
                fill={model.color}
                fillOpacity={0.1}
              />
            ))}
          </RadarChart>
        </ResponsiveContainer>
      </div>
    </motion.section>
  );
}
