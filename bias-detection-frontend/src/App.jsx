import React, { useState, useEffect } from 'react'
import { Button } from '@/components/ui/button.jsx'
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card.jsx'
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs.jsx'
import { Badge } from '@/components/ui/badge.jsx'
import { Progress } from '@/components/ui/progress.jsx'
import { Alert, AlertDescription, AlertTitle } from '@/components/ui/alert.jsx'
import { Textarea } from '@/components/ui/textarea.jsx'
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select.jsx'
import { Label } from '@/components/ui/label.jsx'
import { Input } from '@/components/ui/input.jsx'
import { 
  Shield, 
  AlertTriangle, 
  CheckCircle, 
  BarChart3, 
  Upload, 
  Download,
  Brain,
  Users,
  Scale,
  Eye,
  TrendingUp,
  FileText,
  Settings
} from 'lucide-react'
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, RadarChart, PolarGrid, PolarAngleAxis, PolarRadiusAxis, Radar, LineChart, Line } from 'recharts'
import './App.css'

const API_BASE = '/api/bias'

function App() {
  const [activeTab, setActiveTab] = useState('analyze')
  const [analysisResults, setAnalysisResults] = useState(null)
  const [loading, setLoading] = useState(false)
  const [csvData, setCsvData] = useState('')
  const [modelType, setModelType] = useState('RandomForest')
  const [datasetName, setDatasetName] = useState('Sample Dataset')
  const [analyses, setAnalyses] = useState([])
  const [fairnessMetrics, setFairnessMetrics] = useState([])
  const [mitigationStrategies, setMitigationStrategies] = useState([])

  useEffect(() => {
    fetchAnalyses()
    fetchFairnessMetrics()
    fetchMitigationStrategies()
  }, [])

  const fetchAnalyses = async () => {
    try {
      const response = await fetch(`${API_BASE}/analyses`)
      const data = await response.json()
      setAnalyses(data)
    } catch (error) {
      console.error('Error fetching analyses:', error)
    }
  }

  const fetchFairnessMetrics = async () => {
    try {
      const response = await fetch(`${API_BASE}/metrics`)
      const data = await response.json()
      setFairnessMetrics(data)
    } catch (error) {
      console.error('Error fetching metrics:', error)
    }
  }

  const fetchMitigationStrategies = async () => {
    try {
      const response = await fetch(`${API_BASE}/mitigation-strategies`)
      const data = await response.json()
      setMitigationStrategies(data)
    } catch (error) {
      console.error('Error fetching strategies:', error)
    }
  }

  const generateSampleData = async () => {
    try {
      const response = await fetch(`${API_BASE}/sample-data`)
      const data = await response.json()
      setCsvData(data.csv_data)
      setDatasetName('Generated Sample Dataset')
    } catch (error) {
      console.error('Error generating sample data:', error)
    }
  }

  const runBiasAnalysis = async () => {
    setLoading(true)
    try {
      const payload = {
        use_sample_data: !csvData,
        csv_data: csvData,
        model_type: modelType,
        dataset_name: datasetName
      }

      const response = await fetch(`${API_BASE}/analyze`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(payload)
      })

      const data = await response.json()
      setAnalysisResults(data)
      fetchAnalyses()
    } catch (error) {
      console.error('Error running analysis:', error)
    } finally {
      setLoading(false)
    }
  }

  const getBiasLevelColor = (level) => {
    switch (level) {
      case 'Low': return 'bg-green-100 text-green-800'
      case 'Medium': return 'bg-yellow-100 text-yellow-800'
      case 'High': return 'bg-red-100 text-red-800'
      default: return 'bg-gray-100 text-gray-800'
    }
  }

  const getBiasLevelIcon = (level) => {
    switch (level) {
      case 'Low': return <CheckCircle className="h-4 w-4" />
      case 'Medium': return <AlertTriangle className="h-4 w-4" />
      case 'High': return <AlertTriangle className="h-4 w-4" />
      default: return <Eye className="h-4 w-4" />
    }
  }

  const formatMetricData = (metrics) => {
    if (!metrics) return []
    
    const data = []
    Object.entries(metrics).forEach(([attribute, attributeMetrics]) => {
      Object.entries(attributeMetrics).forEach(([metricName, metricData]) => {
        data.push({
          attribute,
          metric: metricName.replace('_', ' '),
          value: metricData.value,
          threshold: metricData.threshold_acceptable || 0.2
        })
      })
    })
    return data
  }

  return (
    <div className="min-h-screen bg-gradient-to-br from-blue-50 to-indigo-100">
      <div className="container mx-auto px-4 py-8">
        {/* Header */}
        <div className="mb-8">
          <div className="flex items-center gap-3 mb-4">
            <Shield className="h-8 w-8 text-blue-600" />
            <h1 className="text-3xl font-bold text-gray-900">Ethical AI Bias Detection Platform</h1>
          </div>
          <p className="text-gray-600 text-lg">
            Comprehensive bias analysis and fairness assessment for machine learning models
          </p>
        </div>

        <Tabs value={activeTab} onValueChange={setActiveTab} className="space-y-6">
          <TabsList className="grid w-full grid-cols-5">
            <TabsTrigger value="analyze" className="flex items-center gap-2">
              <Brain className="h-4 w-4" />
              Analyze
            </TabsTrigger>
            <TabsTrigger value="results" className="flex items-center gap-2">
              <BarChart3 className="h-4 w-4" />
              Results
            </TabsTrigger>
            <TabsTrigger value="history" className="flex items-center gap-2">
              <FileText className="h-4 w-4" />
              History
            </TabsTrigger>
            <TabsTrigger value="metrics" className="flex items-center gap-2">
              <Scale className="h-4 w-4" />
              Metrics
            </TabsTrigger>
            <TabsTrigger value="strategies" className="flex items-center gap-2">
              <Settings className="h-4 w-4" />
              Strategies
            </TabsTrigger>
          </TabsList>

          {/* Analysis Tab */}
          <TabsContent value="analyze" className="space-y-6">
            <Card>
              <CardHeader>
                <CardTitle className="flex items-center gap-2">
                  <Upload className="h-5 w-5" />
                  Dataset Configuration
                </CardTitle>
                <CardDescription>
                  Upload your dataset or use sample data to perform bias analysis
                </CardDescription>
              </CardHeader>
              <CardContent className="space-y-4">
                <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                  <div className="space-y-2">
                    <Label htmlFor="dataset-name">Dataset Name</Label>
                    <Input
                      id="dataset-name"
                      value={datasetName}
                      onChange={(e) => setDatasetName(e.target.value)}
                      placeholder="Enter dataset name"
                    />
                  </div>
                  <div className="space-y-2">
                    <Label htmlFor="model-type">Model Type</Label>
                    <Select value={modelType} onValueChange={setModelType}>
                      <SelectTrigger>
                        <SelectValue placeholder="Select model type" />
                      </SelectTrigger>
                      <SelectContent>
                        <SelectItem value="RandomForest">Random Forest</SelectItem>
                        <SelectItem value="LogisticRegression">Logistic Regression</SelectItem>
                        <SelectItem value="SVM">Support Vector Machine</SelectItem>
                      </SelectContent>
                    </Select>
                  </div>
                </div>

                <div className="space-y-2">
                  <Label htmlFor="csv-data">CSV Data (Optional)</Label>
                  <Textarea
                    id="csv-data"
                    value={csvData}
                    onChange={(e) => setCsvData(e.target.value)}
                    placeholder="Paste your CSV data here or use sample data"
                    rows={6}
                  />
                </div>

                <div className="flex gap-2">
                  <Button onClick={generateSampleData} variant="outline">
                    Generate Sample Data
                  </Button>
                  <Button onClick={runBiasAnalysis} disabled={loading}>
                    {loading ? 'Analyzing...' : 'Run Bias Analysis'}
                  </Button>
                </div>
              </CardContent>
            </Card>
          </TabsContent>

          {/* Results Tab */}
          <TabsContent value="results" className="space-y-6">
            {analysisResults ? (
              <>
                {/* Overview Cards */}
                <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
                  <Card>
                    <CardContent className="p-6">
                      <div className="flex items-center justify-between">
                        <div>
                          <p className="text-sm font-medium text-gray-600">Overall Accuracy</p>
                          <p className="text-2xl font-bold">{(analysisResults.overall_accuracy * 100).toFixed(1)}%</p>
                        </div>
                        <TrendingUp className="h-8 w-8 text-blue-600" />
                      </div>
                    </CardContent>
                  </Card>

                  <Card>
                    <CardContent className="p-6">
                      <div className="flex items-center justify-between">
                        <div>
                          <p className="text-sm font-medium text-gray-600">Bias Score</p>
                          <p className="text-2xl font-bold">{analysisResults.overall_bias_score.toFixed(3)}</p>
                        </div>
                        <Scale className="h-8 w-8 text-orange-600" />
                      </div>
                    </CardContent>
                  </Card>

                  <Card>
                    <CardContent className="p-6">
                      <div className="flex items-center justify-between">
                        <div>
                          <p className="text-sm font-medium text-gray-600">Bias Level</p>
                          <Badge className={getBiasLevelColor(analysisResults.bias_level)}>
                            {getBiasLevelIcon(analysisResults.bias_level)}
                            {analysisResults.bias_level}
                          </Badge>
                        </div>
                        <AlertTriangle className="h-8 w-8 text-red-600" />
                      </div>
                    </CardContent>
                  </Card>

                  <Card>
                    <CardContent className="p-6">
                      <div className="flex items-center justify-between">
                        <div>
                          <p className="text-sm font-medium text-gray-600">Sample Size</p>
                          <p className="text-2xl font-bold">{analysisResults.sample_size}</p>
                        </div>
                        <Users className="h-8 w-8 text-green-600" />
                      </div>
                    </CardContent>
                  </Card>
                </div>

                {/* Fairness Metrics Chart */}
                <Card>
                  <CardHeader>
                    <CardTitle>Fairness Metrics Analysis</CardTitle>
                    <CardDescription>
                      Bias scores across different protected attributes and fairness metrics
                    </CardDescription>
                  </CardHeader>
                  <CardContent>
                    <ResponsiveContainer width="100%" height={400}>
                      <BarChart data={formatMetricData(analysisResults.fairness_metrics)}>
                        <CartesianGrid strokeDasharray="3 3" />
                        <XAxis dataKey="metric" />
                        <YAxis />
                        <Tooltip />
                        <Bar dataKey="value" fill="#3b82f6" />
                        <Bar dataKey="threshold" fill="#ef4444" opacity={0.3} />
                      </BarChart>
                    </ResponsiveContainer>
                  </CardContent>
                </Card>

                {/* Mitigation Strategies */}
                <Card>
                  <CardHeader>
                    <CardTitle>Recommended Mitigation Strategies</CardTitle>
                    <CardDescription>
                      Suggested approaches to reduce bias in your model
                    </CardDescription>
                  </CardHeader>
                  <CardContent>
                    <div className="space-y-4">
                      {analysisResults.mitigation_strategies.map((strategy, index) => (
                        <Alert key={index}>
                          <Settings className="h-4 w-4" />
                          <AlertTitle>{strategy.name}</AlertTitle>
                          <AlertDescription>
                            <p className="mb-2">{strategy.description}</p>
                            <div className="flex gap-2">
                              <Badge variant="outline">{strategy.type}</Badge>
                              <Badge variant="outline">{strategy.priority} priority</Badge>
                              <Badge variant="outline">{strategy.complexity} complexity</Badge>
                            </div>
                          </AlertDescription>
                        </Alert>
                      ))}
                    </div>
                  </CardContent>
                </Card>
              </>
            ) : (
              <Card>
                <CardContent className="p-12 text-center">
                  <Brain className="h-12 w-12 text-gray-400 mx-auto mb-4" />
                  <h3 className="text-lg font-semibold text-gray-900 mb-2">No Analysis Results</h3>
                  <p className="text-gray-600 mb-4">Run a bias analysis to see results here</p>
                  <Button onClick={() => setActiveTab('analyze')}>
                    Start Analysis
                  </Button>
                </CardContent>
              </Card>
            )}
          </TabsContent>

          {/* History Tab */}
          <TabsContent value="history" className="space-y-6">
            <Card>
              <CardHeader>
                <CardTitle>Analysis History</CardTitle>
                <CardDescription>
                  Previous bias analyses and their results
                </CardDescription>
              </CardHeader>
              <CardContent>
                {analyses.length > 0 ? (
                  <div className="space-y-4">
                    {analyses.map((analysis) => (
                      <div key={analysis.id} className="border rounded-lg p-4">
                        <div className="flex items-center justify-between mb-2">
                          <h4 className="font-semibold">{analysis.dataset_name}</h4>
                          <Badge className={getBiasLevelColor(
                            analysis.bias_score < 0.1 ? 'Low' : 
                            analysis.bias_score < 0.2 ? 'Medium' : 'High'
                          )}>
                            Bias Score: {analysis.bias_score.toFixed(3)}
                          </Badge>
                        </div>
                        <div className="text-sm text-gray-600">
                          <p>Model: {analysis.model_type}</p>
                          <p>Date: {new Date(analysis.created_at).toLocaleDateString()}</p>
                          <p>Protected Attributes: {analysis.protected_attributes.join(', ')}</p>
                        </div>
                      </div>
                    ))}
                  </div>
                ) : (
                  <div className="text-center py-8">
                    <FileText className="h-12 w-12 text-gray-400 mx-auto mb-4" />
                    <p className="text-gray-600">No analysis history available</p>
                  </div>
                )}
              </CardContent>
            </Card>
          </TabsContent>

          {/* Metrics Tab */}
          <TabsContent value="metrics" className="space-y-6">
            <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
              {fairnessMetrics.map((metric, index) => (
                <Card key={index}>
                  <CardHeader>
                    <CardTitle className="flex items-center gap-2">
                      <Scale className="h-5 w-5" />
                      {metric.name}
                    </CardTitle>
                    <CardDescription>{metric.type} fairness metric</CardDescription>
                  </CardHeader>
                  <CardContent>
                    <p className="text-sm text-gray-600 mb-4">{metric.description}</p>
                    <div className="space-y-2">
                      <div className="text-xs font-mono bg-gray-100 p-2 rounded">
                        {metric.formula}
                      </div>
                      <p className="text-sm">{metric.interpretation}</p>
                      <div className="flex gap-2">
                        <Badge variant="outline">Good: &lt; {metric.threshold_good}</Badge>
                        <Badge variant="outline">Acceptable: &lt; {metric.threshold_acceptable}</Badge>
                      </div>
                    </div>
                  </CardContent>
                </Card>
              ))}
            </div>
          </TabsContent>

          {/* Strategies Tab */}
          <TabsContent value="strategies" className="space-y-6">
            <div className="space-y-6">
              {mitigationStrategies.map((strategy, index) => (
                <Card key={index}>
                  <CardHeader>
                    <CardTitle className="flex items-center gap-2">
                      <Settings className="h-5 w-5" />
                      {strategy.name}
                    </CardTitle>
                    <CardDescription>
                      <Badge variant="outline">{strategy.type}</Badge>
                      <Badge variant="outline" className="ml-2">{strategy.complexity} complexity</Badge>
                    </CardDescription>
                  </CardHeader>
                  <CardContent>
                    <p className="text-gray-600 mb-4">{strategy.description}</p>
                    
                    <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                      <div>
                        <h4 className="font-semibold text-sm mb-2">Techniques</h4>
                        <ul className="text-sm text-gray-600 space-y-1">
                          {strategy.techniques.map((technique, i) => (
                            <li key={i}>• {technique}</li>
                          ))}
                        </ul>
                      </div>
                      
                      <div>
                        <h4 className="font-semibold text-sm mb-2">Pros</h4>
                        <ul className="text-sm text-green-600 space-y-1">
                          {strategy.pros.map((pro, i) => (
                            <li key={i}>• {pro}</li>
                          ))}
                        </ul>
                      </div>
                      
                      <div>
                        <h4 className="font-semibold text-sm mb-2">Cons</h4>
                        <ul className="text-sm text-red-600 space-y-1">
                          {strategy.cons.map((con, i) => (
                            <li key={i}>• {con}</li>
                          ))}
                        </ul>
                      </div>
                    </div>
                  </CardContent>
                </Card>
              ))}
            </div>
          </TabsContent>
        </Tabs>
      </div>
    </div>
  )
}

export default App

