'use client';
import { motion } from 'framer-motion';

export default function HeroBanner({ data }: { data: any }) {
  const bestModel = data.models.reduce((best: any, current: any) => 
    current.metrics.auc > best.metrics.auc ? current : best
  );

  return (
    <motion.section 
      className="hero"
      initial={{ opacity: 0, y: 30 }}
      whileInView={{ opacity: 1, y: 0 }}
      viewport={{ once: true, margin: '-100px' }}
      transition={{ duration: 0.6, ease: 'easeOut' }}
    >
      <h1 className="hero-title">
        Diabetes Risk Assessment: <span>ML Classifier Comparison</span>
      </h1>
      <p className="hero-subtitle">
        Evaluating {data.models.length} machine learning models for diabetes prediction using the {data.dataset_info.name}.
      </p>
      
      <div className="stats-grid">
        <div className="stat-card">
          <div className="stat-value">{data.dataset_info.total_samples}</div>
          <div className="stat-label">Total Samples</div>
        </div>
        <div className="stat-card">
          <div className="stat-value">{data.dataset_info.features.length}</div>
          <div className="stat-label">Features Analyzed</div>
        </div>
        <div className="stat-card">
          <div className="stat-value">{bestModel.metrics.auc.toFixed(4)}</div>
          <div className="stat-label">Best AUC ({bestModel.short_name})</div>
        </div>
      </div>
    </motion.section>
  );
}
