'use client';
import resultsData from '@/lib/data';
import HeroBanner from '@/components/HeroBanner';
import ModelComparison from '@/components/ModelComparison';
import ROCCurves from '@/components/ROCCurves';
import ConfusionMatrices from '@/components/ConfusionMatrices';
import FeatureImportance from '@/components/FeatureImportance';
import CrossValidation from '@/components/CrossValidation';
import RadarComparison from '@/components/RadarComparison';
import ResultsTable from '@/components/ResultsTable';
import Footer from '@/components/Footer';

export default function Home() {
  return (
    <main className="dashboard">
      <HeroBanner data={resultsData} />
      <ModelComparison data={resultsData} />
      <ROCCurves data={resultsData} />
      <ConfusionMatrices data={resultsData} />
      <FeatureImportance data={resultsData} />
      <CrossValidation data={resultsData} />
      <RadarComparison data={resultsData} />
      <ResultsTable data={resultsData} />
      <Footer />
    </main>
  );
}
