'use client';
import { motion } from 'framer-motion';
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer } from 'recharts';

export default function ModelComparison({ data }: { data: any }) {
  const chartData = [
    { name: 'Balanced Accuracy', ...Object.fromEntries(data.models.map((m: any) => [m.short_name, m.metrics.balanced_accuracy])) },
    { name: 'Accuracy', ...Object.fromEntries(data.models.map((m: any) => [m.short_name, m.metrics.accuracy])) },
    { name: 'Precision', ...Object.fromEntries(data.models.map((m: any) => [m.short_name, m.metrics.precision])) },
    { name: 'Recall', ...Object.fromEntries(data.models.map((m: any) => [m.short_name, m.metrics.recall])) },
    { name: 'F1 Score', ...Object.fromEntries(data.models.map((m: any) => [m.short_name, m.metrics.f1_score])) },
    { name: 'AUC', ...Object.fromEntries(data.models.map((m: any) => [m.short_name, m.metrics.auc])) },
  ];

  return (
    <motion.section 
      className="section"
      initial={{ opacity: 0, y: 30 }}
      whileInView={{ opacity: 1, y: 0 }}
      viewport={{ once: true, margin: '-100px' }}
      transition={{ duration: 0.6, ease: 'easeOut' }}
    >
      <h2 className="section-title">Model Performance Comparison</h2>
      <p className="section-subtitle">Class-aware and threshold-based metrics across all tested classifiers</p>
      
      <div className="chart-container chart-container-lg">
        <ResponsiveContainer width="100%" height="100%">
          <BarChart data={chartData} margin={{ top: 20, right: 30, left: 20, bottom: 5 }}>
            <CartesianGrid strokeDasharray="3 3" stroke="rgba(148, 163, 184, 0.1)" />
            <XAxis dataKey="name" stroke="#94a3b8" />
            <YAxis domain={[0, 1]} stroke="#94a3b8" />
            <Tooltip contentStyle={{ backgroundColor: 'rgba(15, 23, 42, 0.9)', borderColor: 'rgba(148, 163, 184, 0.2)', borderRadius: '8px' }} />
            <Legend />
            {data.models.map((model: any) => (
              <Bar key={model.short_name} dataKey={model.short_name} name={model.name} fill={model.color} radius={[4, 4, 0, 0]} />
            ))}
          </BarChart>
        </ResponsiveContainer>
      </div>
    </motion.section>
  );
}
