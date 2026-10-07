'use client';
import { motion } from 'framer-motion';

export default function HeroBanner({ data }: { data: any }) {
  const bestModel = data.models.reduce((best: any, current: any) =>
    current.metrics.auc > best.metrics.auc ? current : best
  );
  const positive = data.dataset_info.class_distribution.positive;
  const negative = data.dataset_info.class_distribution.negative;
  const positiveShare = positive / data.dataset_info.total_samples;
  const negativeShare = negative / data.dataset_info.total_samples;
  const modelStrengths: Record<string, string> = {
    LR: 'Interpretable baseline',
    RF: 'Best overall AUC',
    SVM: 'Strong margin separation',
    KNN: 'Fast local decisions',
    XGB: 'High-capacity boosting',
  };

  return (
    <motion.section
      className="overview"
      initial={{ opacity: 0, y: 30 }}
      whileInView={{ opacity: 1, y: 0 }}
      viewport={{ once: true, margin: '-100px' }}
      transition={{ duration: 0.6, ease: 'easeOut' }}
    >
      <header className="console-header">
        <div className="brand-lockup">
          <img className="brand-logo" src="/data/logo.png" alt="Diabetes Risk Lab logo" />
          <div>
            <strong>Diabetes Risk Lab</strong>
            <span>classifier observability console</span>
          </div>
        </div>
        <div className="header-status"><span className="status-dot" /> Evaluation complete <span className="header-divider">/</span> 5 models <span className="header-time">seed 42</span></div>
      </header>

      <div className="overview-heading">
        <div>
          <p className="eyebrow">Model comparison / binary classification</p>
          <h1>Diabetes mellitus risk assessment</h1>
          <p className="overview-copy">A reproducible benchmark of five classifiers on diagnostic measurements, with the emphasis on discrimination, stability, and the cost of missed positive cases.</p>
        </div>
        <div className="run-summary">
          <span>Best discriminator</span>
          <strong style={{ color: bestModel.color }}>{bestModel.short_name} <em>{bestModel.metrics.auc.toFixed(4)} AUC</em></strong>
          <small>held-out test set / stratified split</small>
        </div>
      </div>

      <div className="stats-grid">
        <div className="stat-card"><span className="stat-kicker">Dataset rows</span><strong className="stat-value">{data.dataset_info.total_samples}</strong><span className="stat-label">{data.dataset_info.train_size} train / {data.dataset_info.test_size} test</span></div>
        <div className="stat-card"><span className="stat-kicker">Input signals</span><strong className="stat-value">{data.dataset_info.features.length}</strong><span className="stat-label">clinical features analyzed</span></div>
        <div className="stat-card"><span className="stat-kicker">Positive outcome</span><strong className="stat-value stat-value-warm">{(positiveShare * 100).toFixed(1)}%</strong><span className="stat-label">{positive} diabetes cases</span></div>
        <div className="stat-card"><span className="stat-kicker">Negative outcome</span><strong className="stat-value stat-value-green">{(negativeShare * 100).toFixed(1)}%</strong><span className="stat-label">{negative} non-diabetes cases</span></div>
      </div>

      <div className="overview-grid">
        <article className="info-panel dataset-panel">
          <div className="panel-heading"><div><span className="panel-index">01</span><h2>Dataset profile</h2></div><span className="panel-tag">CDC / BRFSS 2015</span></div>
          <p>{data.dataset_info.description}</p>
          <div className="balance-row"><div><span>Class balance</span><strong>{positiveShare < 0.4 ? 'Imbalanced' : 'Balanced'}</strong></div><div className="balance-bar"><span style={{ width: `${negativeShare * 100}%` }} /><i style={{ width: `${positiveShare * 100}%` }} /></div><div className="balance-legend"><span><b className="legend-negative" /> No diabetes {negative}</span><span><b className="legend-positive" /> Diabetes {positive}</span></div></div>
          <div className="dataset-meta"><div><span>Target</span><strong>{data.dataset_info.target}</strong></div><div><span>Split</span><strong>80 / 20 stratified</strong></div><div><span>Scaling</span><strong>StandardScaler</strong></div><div><span>Sampling</span><strong>10% / seed 42</strong></div></div>
        </article>

        <article className="info-panel models-panel">
          <div className="panel-heading"><div><span className="panel-index">02</span><h2>Model roster</h2></div><span className="panel-tag">5 estimators</span></div>
          <div className="model-roster">{data.models.map((model: any) => <div className="model-row" key={model.short_name}><span className="model-color" style={{ backgroundColor: model.color }} /><div><strong>{model.name} <small>{model.short_name}</small></strong><span>{model.description}</span></div><em>{modelStrengths[model.short_name]}</em><b>{model.metrics.auc.toFixed(3)}</b></div>)}</div>
        </article>
      </div>

      <div className="feature-strip"><span className="panel-index">03</span><strong>Features used</strong>{data.dataset_info.features.map((feature: any) => <span className="feature-chip" key={feature.name}>{feature.name}</span>)}</div>
    </motion.section>
  );
}
