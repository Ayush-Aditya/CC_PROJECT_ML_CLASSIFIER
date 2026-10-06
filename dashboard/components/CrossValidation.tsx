'use client';
import { motion } from 'framer-motion';
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Cell, ErrorBar } from 'recharts';

export default function CrossValidation({ data }: { data: any }) {
  const calcStats = (arr: number[]) => {
    const mean = arr.reduce((a, b) => a + b, 0) / arr.length;
    const std = Math.sqrt(arr.reduce((a, b) => a + Math.pow(b - mean, 2), 0) / arr.length);
    return { mean, std };
  };

  const chartData = data.models.map((model: any) => {
    const stats = calcStats(model.cv_scores);
    return {
      name: model.short_name,
      fullName: model.name,
      mean: Math.round(stats.mean * 10000) / 10000,
      std: Math.round(stats.std * 10000) / 10000,
      color: model.color,
      scores: model.cv_scores,
    };
  });

  return (
    <motion.section
      className="section"
      initial={{ opacity: 0, y: 30 }}
      whileInView={{ opacity: 1, y: 0 }}
      viewport={{ once: true, margin: '-100px' }}
      transition={{ duration: 0.6, ease: 'easeOut' }}
    >
      <h2 className="section-title">Cross-Validation Stability</h2>
      <p className="section-subtitle">
        5-fold stratified cross-validation accuracy — Mean ± Standard Deviation
      </p>

      <div className="chart-container" style={{ marginBottom: '2rem' }}>
        <ResponsiveContainer width="100%" height="100%">
          <BarChart data={chartData} margin={{ top: 20, right: 30, left: 20, bottom: 5 }}>
            <CartesianGrid strokeDasharray="3 3" stroke="rgba(148, 163, 184, 0.1)" />
            <XAxis dataKey="name" stroke="#94a3b8" />
            <YAxis
              domain={[0.65, 0.9]}
              stroke="#94a3b8"
              tickFormatter={(v: number) => `${(v * 100).toFixed(0)}%`}
            />
            <Tooltip
              contentStyle={{
                backgroundColor: 'rgba(15, 23, 42, 0.95)',
                borderColor: 'rgba(148, 163, 184, 0.2)',
                borderRadius: '8px',
                color: '#f1f5f9'
              }}
              formatter={(value: number, name: string) => {
                if (name === 'mean') return [`${(value * 100).toFixed(2)}%`, 'CV Mean'];
                return [value, name];
              }}
              labelFormatter={(label: string) => {
                const item = chartData.find((d: any) => d.name === label);
                return item ? item.fullName : label;
              }}
            />
            <Bar dataKey="mean" radius={[6, 6, 0, 0]} barSize={50}>
              {chartData.map((entry: any, index: number) => (
                <Cell key={index} fill={entry.color} fillOpacity={0.8} />
              ))}
              <ErrorBar dataKey="std" stroke="#94a3b8" strokeWidth={2} />
            </Bar>
          </BarChart>
        </ResponsiveContainer>
      </div>

      {/* Stats cards below chart */}
      <div className="cv-grid">
        {data.models.map((model: any) => {
          const stats = calcStats(model.cv_scores);
          return (
            <div key={model.short_name} className="cv-card" style={{ borderColor: `color-mix(in srgb, ${model.color} 30%, transparent)` }}>
              <div className="cv-model-name" style={{ color: model.color }}>{model.short_name}</div>
              <div className="cv-mean">{(stats.mean * 100).toFixed(1)}%</div>
              <div className="cv-std">± {(stats.std * 100).toFixed(2)}%</div>
              <div style={{ fontSize: '0.7rem', color: '#64748b', marginTop: '0.5rem' }}>
                Folds: {model.cv_scores.map((s: number) => `${(s * 100).toFixed(1)}`).join(' | ')}
              </div>
            </div>
          );
        })}
      </div>
    </motion.section>
  );
}
