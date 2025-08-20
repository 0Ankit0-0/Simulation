"use client"
import { useState } from "react"
import { ThemeProvider } from "../../components/provider/themeProvider"
import { Card, CardContent, CardHeader, CardTitle } from "../../components/ui/card"
import { Button } from "../ui/button"
import { Badge } from "../ui/badge"
import { Upload, FileText, Loader2, Check, ArrowRight, ArrowLeft } from "lucide-react"
import axios from "axios"

// Backend TODO: Create API endpoints for case creation
// POST /api/cases - Create new case
// POST /api/cases/upload-evidence - Upload evidence files (multipart/form-data)
// POST /api/ai/summarize-evidence - Send evidence to AI for summarization
// PUT /api/cases/{case_id}/ai-summaries - Update AI summaries after user edits
// POST /api/cases/{case_id}/start-simulation - Initialize courtroom simulation
// GET /api/cases/{case_id}/simulation-status - Check simulation readiness

// Flask Backend AI Integration Required:
// - OpenAI/Claude API integration for evidence summarization
// - File storage service (AWS S3, Google Cloud Storage, or local storage)
// - Background job processing (Celery) for AI tasks
// - WebSocket support for real-time processing updates

const CASE_TYPES = [
  "Criminal Law",
  "Civil Law",
  "Corporate Law",
  "Family Law",
  "Property Law",
  "Constitutional Law",
  "Contract Law",
  "Tort Law",
]

export default function CasePage() {
  const [currentStep, setCurrentStep] = useState(1) // 1: Form, 2: Loading, 3: AI Summary, 4: Confirmation
  const [isLoading, setIsLoading] = useState(false)
  const [caseId, setCaseId] = useState(null)

  // Form data
  const [caseData, setCaseData] = useState({
    title: "",
    description: "",
    type: "",
    evidence: [],
  })

  // AI generated summaries (mock data for now)
  const [aiSummaries, setAiSummaries] = useState([])

  const handleInputChange = (field, value) => {
    setCaseData((prev) => ({ ...prev, [field]: value }))
  }

  const handleFileUpload = async (event) => {
    const files = Array.from(event.target.files)
    const formData = new FormData()

    files.forEach((file) => {
      formData.append("files", file)
    })

    try {
      const response = await axios.post("/api/cases/upload-evidence", formData, {
        headers: {
          "Content-Type": "multipart/form-data",
        },
      })

      const newEvidence = response.data.map((file) => ({
        id: file.file_id,
        file: null,
        name: file.original_name,
        size: file.file_size,
        type: file.file_type,
        aiSummary: file.ai_summary,
      }))

      setCaseData((prev) => ({
        ...prev,
        evidence: [...prev.evidence, ...newEvidence],
      }))
    } catch (error) {
      console.error("Error uploading evidence:", error)
    }
  }

  const removeEvidence = (id) => {
    setCaseData((prev) => ({
      ...prev,
      evidence: prev.evidence.filter((item) => item.id !== id),
    }))
  }

  const handleNext = async () => {
    if (currentStep === 1) {
      // Create new case
      try {
        const response = await axios.post("/api/cases", caseData)
        setCaseId(response.data._id)
        setCurrentStep(2)
        setIsLoading(true)
      } catch (error) {
        console.error("Error creating case:", error)
      }
    } else if (currentStep === 2) {
      // Start AI processing
      try {
        const response = await axios.post("/api/ai/summarize-evidence", {
          caseId: caseId,
          evidenceFiles: caseData.evidence.map((item) => item.id),
        })

        setAiSummaries(response.data.aiSummaries)
        setIsLoading(false)
        setCurrentStep(3)
      } catch (error) {
        console.error("Error processing AI summaries:", error)
      }
    } else if (currentStep === 3) {
      // Update AI summaries
      try {
        await axios.put(`/api/cases/${caseId}/ai-summaries`, {
          aiSummaries: aiSummaries,
        })
        setCurrentStep(4)
      } catch (error) {
        console.error("Error updating AI summaries:", error)
      }
    } else if (currentStep === 4) {
      // Start simulation
      try {
        await axios.post(`/api/cases/${caseId}/start-simulation`)
        console.log("Starting simulation with:", { caseData, aiSummaries })
        // Redirect to simulation page
        window.location.href = `/simulation/${caseId}`
      } catch (error) {
        console.error("Error starting simulation:", error)
      }
    }
  }

  const handleBack = () => {
    if (currentStep > 1) {
      setCurrentStep(currentStep - 1)
    }
  }

  const updateAiSummary = (id, newSummary) => {
    setAiSummaries((prev) => prev.map((item) => (item.id === id ? { ...item, aiSummary: newSummary } : item)))
  }

  const isFormValid = caseData.title && caseData.description && caseData.type && caseData.evidence.length > 0

  return (
    <ThemeProvider attribute="class" defaultTheme="system" enableSystem disableTransitionOnChange>
      <div className="min-h-screen bg-background">
        <main className="container mx-auto px-4 py-8">
          <div className="max-w-4xl mx-auto">
            {/* Step 1: Case Details Form */}
            {currentStep === 1 && (
              <Card>
                <CardHeader>
                  <CardTitle>Create New Case</CardTitle>
                  <p className="text-muted-foreground">
                    Enter the details of your case and upload relevant evidence files
                  </p>
                </CardHeader>
                <CardContent className="space-y-6">
                  {/* Case Title */}
                  <div>
                    <label className="block text-sm font-medium mb-2">Case Title *</label>
                    <input
                      type="text"
                      value={caseData.title}
                      onChange={(e) => handleInputChange("title", e.target.value)}
                      placeholder="Enter case title..."
                      className="w-full px-3 py-2 border rounded-md bg-background focus:outline-none focus:ring-2 focus:ring-primary"
                    />
                  </div>

                  {/* Case Description */}
                  <div>
                    <label className="block text-sm font-medium mb-2">Case Description *</label>
                    <textarea
                      value={caseData.description}
                      onChange={(e) => handleInputChange("description", e.target.value)}
                      placeholder="Provide a detailed description of the case..."
                      rows={4}
                      className="w-full px-3 py-2 border rounded-md bg-background focus:outline-none focus:ring-2 focus:ring-primary resize-none"
                    />
                  </div>

                  {/* Case Type */}
                  <div>
                    <label className="block text-sm font-medium mb-2">Case Type *</label>
                    <select
                      value={caseData.type}
                      onChange={(e) => handleInputChange("type", e.target.value)}
                      className="w-full px-3 py-2 border rounded-md bg-background focus:outline-none focus:ring-2 focus:ring-primary"
                    >
                      <option value="">Select case type...</option>
                      {CASE_TYPES.map((type) => (
                        <option key={type} value={type}>
                          {type}
                        </option>
                      ))}
                    </select>
                  </div>

                  {/* Evidence Upload */}
                  <div>
                    <label className="block text-sm font-medium mb-2">Evidence Files *</label>
                    <div className="border-2 border-dashed border-muted-foreground/25 rounded-lg p-6 text-center">
                      <Upload className="h-12 w-12 mx-auto text-muted-foreground mb-4" />
                      <p className="text-muted-foreground mb-4">
                        Upload evidence files (documents, images, audio, video)
                      </p>
                      <input
                        type="file"
                        multiple
                        onChange={handleFileUpload}
                        className="hidden"
                        id="evidence-upload"
                        accept=".pdf,.doc,.docx,.txt,.jpg,.jpeg,.png,.mp3,.mp4,.wav"
                      />
                      <Button asChild variant="outline">
                        <label htmlFor="evidence-upload" className="cursor-pointer">
                          Choose Files
                        </label>
                      </Button>
                    </div>

                    {/* Uploaded Files */}
                    {caseData.evidence.length > 0 && (
                      <div className="mt-4 space-y-2">
                        <p className="text-sm font-medium">Uploaded Files:</p>
                        {caseData.evidence.map((item) => (
                          <div key={item.id} className="flex items-center justify-between p-3 border rounded-md">
                            <div className="flex items-center gap-3">
                              <FileText className="h-5 w-5 text-muted-foreground" />
                              <div>
                                <p className="font-medium">{item.name}</p>
                                <p className="text-sm text-muted-foreground">
                                  {(item.size / 1024 / 1024).toFixed(2)} MB
                                </p>
                              </div>
                            </div>
                            <Button
                              variant="ghost"
                              size="sm"
                              onClick={() => removeEvidence(item.id)}
                              className="text-destructive hover:text-destructive"
                            >
                              Remove
                            </Button>
                          </div>
                        ))}
                      </div>
                    )}
                  </div>

                  <div className="flex justify-end">
                    <Button onClick={handleNext} disabled={!isFormValid} className="flex items-center gap-2">
                      Next <ArrowRight className="h-4 w-4" />
                    </Button>
                  </div>
                </CardContent>
              </Card>
            )}

            {/* Step 2: Loading Screen */}
            {currentStep === 2 && (
              <Card>
                <CardContent className="py-16 text-center">
                  <Loader2 className="h-16 w-16 animate-spin mx-auto text-primary mb-6" />
                  <h2 className="text-2xl font-bold mb-4">Processing Evidence</h2>
                  <p className="text-muted-foreground mb-6">
                    Our AI is analyzing and summarizing your evidence files...
                  </p>
                  <div className="max-w-md mx-auto">
                    <div className="bg-muted rounded-full h-2 overflow-hidden">
                      <div className="bg-primary h-full w-2/3 animate-pulse"></div>
                    </div>
                  </div>
                </CardContent>
              </Card>
            )}

            {/* Step 3: AI Summary Review */}
            {currentStep === 3 && (
              <Card>
                <CardHeader>
                  <CardTitle>Review AI-Generated Summaries</CardTitle>
                  <p className="text-muted-foreground">
                    Review and edit the AI-generated summaries for your evidence files
                  </p>
                </CardHeader>
                <CardContent className="space-y-6">
                  {/* Case Summary */}
                  <div className="p-4 border rounded-lg bg-muted/50">
                    <h3 className="font-semibold mb-2">Case Overview</h3>
                    <p>
                      <strong>Title:</strong> {caseData.title}
                    </p>
                    <p>
                      <strong>Type:</strong> {caseData.type}
                    </p>
                    <p>
                      <strong>Description:</strong> {caseData.description}
                    </p>
                    <p>
                      <strong>Evidence Files:</strong> {caseData.evidence.length} files uploaded
                    </p>
                  </div>

                  {/* AI Summaries */}
                  {aiSummaries.map((summary) => (
                    <div key={summary.id} className="border rounded-lg p-4">
                      <div className="flex items-center justify-between mb-3">
                        <h4 className="font-medium flex items-center gap-2">
                          <FileText className="h-4 w-4" />
                          {summary.fileName}
                        </h4>
                        <Badge variant="secondary">AI Generated</Badge>
                      </div>

                      <div className="space-y-3">
                        <div>
                          <label className="block text-sm font-medium mb-1">
                            AI Summary (Max {summary.maxLength} characters)
                          </label>
                          <textarea
                            value={summary.aiSummary}
                            onChange={(e) => updateAiSummary(summary.id, e.target.value)}
                            maxLength={summary.maxLength}
                            rows={4}
                            className="w-full px-3 py-2 border rounded-md bg-background focus:outline-none focus:ring-2 focus:ring-primary resize-none"
                          />
                          <p className="text-xs text-muted-foreground mt-1">
                            {summary.aiSummary.length}/{summary.maxLength} characters
                          </p>
                        </div>
                      </div>
                    </div>
                  ))}

                  <div className="flex justify-between">
                    <Button variant="outline" onClick={handleBack} className="flex items-center gap-2 bg-transparent">
                      <ArrowLeft className="h-4 w-4" /> Back
                    </Button>
                    <Button onClick={handleNext} className="flex items-center gap-2">
                      Continue <ArrowRight className="h-4 w-4" />
                    </Button>
                  </div>
                </CardContent>
              </Card>
            )}

            {/* Step 4: Confirmation */}
            {currentStep === 4 && (
              <Card>
                <CardHeader>
                  <CardTitle className="flex items-center gap-2">
                    <Check className="h-6 w-6 text-green-500" />
                    Case Ready for Simulation
                  </CardTitle>
                  <p className="text-muted-foreground">Your case has been processed and is ready to begin simulation</p>
                </CardHeader>
                <CardContent className="space-y-6">
                  <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                    <div className="space-y-4">
                      <h3 className="font-semibold">Case Details</h3>
                      <div className="space-y-2 text-sm">
                        <p>
                          <strong>Title:</strong> {caseData.title}
                        </p>
                        <p>
                          <strong>Type:</strong> {caseData.type}
                        </p>
                        <p>
                          <strong>Evidence Files:</strong> {caseData.evidence.length}
                        </p>
                        <p>
                          <strong>AI Summaries:</strong> Generated and reviewed
                        </p>
                      </div>
                    </div>

                    <div className="space-y-4">
                      <h3 className="font-semibold">What's Next?</h3>
                      <ul className="space-y-2 text-sm text-muted-foreground">
                        <li>• AI will generate realistic courtroom scenarios</li>
                        <li>• You'll interact with AI judges, lawyers, and witnesses</li>
                        <li>• Practice your legal arguments and procedures</li>
                        <li>• Receive detailed feedback on your performance</li>
                      </ul>
                    </div>
                  </div>

                  <div className="flex justify-between">
                    <Button variant="outline" onClick={handleBack} className="flex items-center gap-2 bg-transparent">
                      <ArrowLeft className="h-4 w-4" /> Back to Review
                    </Button>
                    <Button onClick={handleNext} size="lg" className="flex items-center gap-2">
                      Start Simulation <ArrowRight className="h-4 w-4" />
                    </Button>
                  </div>
                </CardContent>
              </Card>
            )}
          </div>
        </main>
      </div>
    </ThemeProvider>
  )
}
