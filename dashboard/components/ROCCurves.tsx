'use client';
import { motion } from 'framer-motion';
import {
  LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip,
  Legend, ResponsiveContainer, ReferenceLine
} from 'recharts';

export default function ROCCurves({ data }: { data: any }) {
  // Merge all ROC curves by interpolating onto common FPR grid
  const gridSize = 101;
  const commonFprs = Array.from({ length: gridSize }, (_, i) => i / (gridSize - 1));

  const chartData = commonFprs.map(fpr => {
    const point: any = { fpr: Math.round(fpr * 1000) / 1000 };

    data.models.forEach((m: any) => {
      // Interpolate TPR at this FPR
      const curve = m.roc_curve;
      let tpr = 0;
      for (let i = 0; i < curve.length - 1; i++) {
        if (fpr >= curve[i].fpr && fpr <= curve[i + 1].fpr) {
          const frac = curve[i + 1].fpr === curve[i].fpr
            ? 0
            : (fpr - curve[i].fpr) / (curve[i + 1].fpr - curve[i].fpr);
          tpr = curve[i].tpr + frac * (curve[i + 1].tpr - curve[i].tpr);
          break;
        }
      }
      if (fpr >= curve[curve.length - 1].fpr) tpr = curve[curve.length - 1].tpr;
      point[m.short_name] = Math.round(tpr * 10000) / 10000;
    });

    // Diagonal reference line
    point['random'] = Math.round(fpr * 1000) / 1000;
    return point;
  });

  return (
    <motion.section
      className="section"
      initial={{ opacity: 0, y: 30 }}
      whileInView={{ opacity: 1, y: 0 }}
      viewport={{ once: true, margin: '-100px' }}
      transition={{ duration: 0.6, ease: 'easeOut' }}
    >
      <h2 className="section-title">ROC Curves</h2>
      <p className="section-subtitle">
        Receiver Operating Characteristic — higher AUC indicates better discrimination
      </p>

      <div className="chart-container chart-container-lg">
        <ResponsiveContainer width="100%" height="100%">
          <LineChart data={chartData} margin={{ top: 20, right: 30, left: 20, bottom: 30 }}>
            <CartesianGrid strokeDasharray="3 3" stroke="rgba(148, 163, 184, 0.1)" />
            <XAxis
              dataKey="fpr"
              type="number"
              domain={[0, 1]}
              stroke="#94a3b8"
              label={{ value: 'False Positive Rate', position: 'insideBottom', offset: -15, fill: '#94a3b8' }}
              tickCount={6}
            />
            <YAxis
              type="number"
              domain={[0, 1]}
              stroke="#94a3b8"
              label={{ value: 'True Positive Rate', angle: -90, position: 'insideLeft', offset: 10, fill: '#94a3b8' }}
              tickCount={6}
            />
            <Tooltip
              contentStyle={{
                backgroundColor: 'rgba(15, 23, 42, 0.95)',
                borderColor: 'rgba(148, 163, 184, 0.2)',
                borderRadius: '8px',
                color: '#f1f5f9'
              }}
              formatter={(value: number) => value.toFixed(4)}
            />
            <Legend verticalAlign="top" height={40} />

            {/* Diagonal reference line (random classifier) */}
            <Line
              type="linear"
              dataKey="random"
              name="Random (AUC: 0.500)"
              stroke="#475569"
              strokeDasharray="8 4"
              strokeWidth={1.5}
              dot={false}
              isAnimationActive={false}
            />

            {/* Model ROC curves */}
            {data.models.map((model: any) => (
              <Line
                key={model.short_name}
                type="monotone"
                dataKey={model.short_name}
                name={`${model.name} (AUC: ${model.metrics.auc.toFixed(3)})`}
                stroke={model.color}
                strokeWidth={2.5}
                dot={false}
                activeDot={{ r: 4, stroke: model.color, strokeWidth: 2, fill: '#0a0e27' }}
              />
            ))}
          </LineChart>
        </ResponsiveContainer>
      </div>
    </motion.section>
  );
}
