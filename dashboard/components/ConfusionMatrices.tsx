'use client';
import { motion } from 'framer-motion';

export default function ConfusionMatrices({ data }: { data: any }) {
  const getMaxVal = (cm: any) => Math.max(cm.tn, cm.fp, cm.fn, cm.tp);
  
  const getCellColor = (val: number, max: number, color: string) => {
    const intensity = Math.max(0.1, val / max);
    return `color-mix(in srgb, ${color} ${intensity * 100}%, transparent)`;
  };

  return (
    <motion.section 
      className="section"
      initial={{ opacity: 0, y: 30 }}
      whileInView={{ opacity: 1, y: 0 }}
      viewport={{ once: true, margin: '-100px' }}
      transition={{ duration: 0.6, ease: 'easeOut' }}
    >
      <h2 className="section-title">Confusion Matrices</h2>
      <p className="section-subtitle">True/False Positives & Negatives across models</p>
      
      <div className="cm-grid">
        {data.models.map((model: any) => {
          const cm = model.confusion_matrix;
          const max = getMaxVal(cm);
          return (
            <div key={model.short_name} className="cm-card">
              <h3 className="cm-card-title" style={{ color: model.color }}>{model.name}</h3>
              <div className="cm-matrix">
                <div className="cm-cell" style={{ backgroundColor: getCellColor(cm.tn, max, model.color) }}>{cm.tn}</div>
                <div className="cm-cell" style={{ backgroundColor: getCellColor(cm.fp, max, model.color) }}>{cm.fp}</div>
                <div className="cm-cell" style={{ backgroundColor: getCellColor(cm.fn, max, model.color) }}>{cm.fn}</div>
                <div className="cm-cell" style={{ backgroundColor: getCellColor(cm.tp, max, model.color) }}>{cm.tp}</div>
              </div>
              <div className="cm-label-row">
                <span className="cm-label">TN / FP</span>
                <span className="cm-label">|</span>
                <span className="cm-label">FN / TP</span>
              </div>
            </div>
          );
        })}
      </div>
    </motion.section>
  );
}
