'use client';
import { motion } from 'framer-motion';

export default function ResultsTable({ data }: { data: any }) {
  const metrics = ['balanced_accuracy', 'accuracy', 'precision', 'recall', 'f1_score', 'auc', 'specificity', 'training_time_ms'];
  
  // Find best value for each metric
  const bestVals: Record<string, number> = {};
  metrics.forEach(metric => {
    const vals = data.models.map((m: any) => m.metrics[metric]);
    bestVals[metric] = metric === 'training_time_ms' ? Math.min(...vals) : Math.max(...vals);
  });

  return (
    <motion.section 
      className="section"
      initial={{ opacity: 0, y: 30 }}
      whileInView={{ opacity: 1, y: 0 }}
      viewport={{ once: true, margin: '-100px' }}
      transition={{ duration: 0.6, ease: 'easeOut' }}
    >
      <h2 className="section-title">Detailed Results Table</h2>
      <p className="section-subtitle">Comprehensive metrics for all evaluated models</p>
      
      <div className="results-table-wrapper">
        <table className="results-table">
          <thead>
            <tr>
              <th>Model</th>
              <th>Balanced Accuracy</th>
              <th>Accuracy</th>
              <th>Precision</th>
              <th>Recall</th>
              <th>F1 Score</th>
              <th>AUC</th>
              <th>Specificity</th>
              <th>Train Time (ms)</th>
            </tr>
          </thead>
          <tbody>
            {data.models.map((model: any) => (
              <tr key={model.short_name}>
                <td>
                  <span className="model-indicator" style={{ backgroundColor: model.color }}></span>
                  {model.name}
                </td>
                {metrics.map(metric => {
                  const val = model.metrics[metric];
                  const isBest = metric !== 'accuracy' && val === bestVals[metric];
                  const displayVal = metric === 'training_time_ms' ? val.toFixed(1) : val.toFixed(4);
                  
                  return (
                    <td key={metric} className={isBest ? 'best-value' : ''}>
                      {displayVal} {isBest && '★'}
                    </td>
                  );
                })}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </motion.section>
  );
}
