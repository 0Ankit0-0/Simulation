"use client"
import { useState, useEffect } from "react"
import { Search, TrendingUp, Users, FileText, MessageSquare } from "lucide-react"

import CaseCard from "../components/homepageComponents/card"

const mockCases = [
    {
        id: 1,
        title: "Corporate Fraud Investigation",
        description: "A complex case involving financial misconduct and securities fraud in a major corporation.",
        type: "Criminal",
        difficulty: 4,
        likes: 45,
        comments: 12,
        publishedBy: "Sarah Johnson",
        publisherRole: "Senior Attorney",
        publisherAvatar: "/professional-woman-lawyer.png",
        created_at: "2024-01-15",
    },
    {
        id: 2,
        title: "Contract Dispute Resolution",
        description: "A commercial contract dispute involving breach of terms and damages calculation.",
        type: "Contract Law",
        difficulty: 3,
        likes: 32,
        comments: 8,
        publishedBy: "Michael Chen",
        publisherRole: "Contract Specialist",
        publisherAvatar: "/professional-lawyer.png",
        created_at: "2024-01-20",
    },
    {
        id: 3,
        title: "Intellectual Property Theft",
        description: "A case involving stolen trade secrets and patent infringement in the tech industry.",
        type: "IP Law",
        difficulty: 5,
        likes: 67,
        comments: 23,
        publishedBy: "Dr. Emily Rodriguez",
        publisherRole: "IP Law Professor",
        publisherAvatar: "/professional-woman-professor.png",
        created_at: "2024-02-01",
    },
    {
        id: 4,
        title: "Employment Discrimination Case",
        description: "A workplace discrimination case involving wrongful termination and civil rights violations.",
        type: "Employment Law",
        difficulty: 3,
        likes: 28,
        comments: 15,
        publishedBy: "James Wilson",
        publisherRole: "Employment Attorney",
        publisherAvatar: "/professional-attorney.png",
        created_at: "2024-02-05",
    },
    {
        id: 5,
        title: "Environmental Law Violation",
        description: "A case involving environmental regulations and corporate responsibility for pollution.",
        type: "Environmental Law",
        difficulty: 4,
        likes: 41,
        comments: 19,
        publishedBy: "Lisa Park",
        publisherRole: "Environmental Lawyer",
        publisherAvatar: "/environmental-lawyer-woman.png",
        created_at: "2024-02-10",
    },
    {
        id: 6,
        title: "Family Custody Battle",
        description: "A complex child custody case involving interstate jurisdiction and parental rights.",
        type: "Family Law",
        difficulty: 2,
        likes: 19,
        comments: 7,
        publishedBy: "Robert Martinez",
        publisherRole: "Family Law Attorney",
        publisherAvatar: "/professional-family-lawyer.png",
        created_at: "2024-02-15",
    },
]

export default function HomePage() {
    const [searchTerm, setSearchTerm] = useState("")
    const [filteredCases, setFilteredCases] = useState(mockCases)

    useEffect(() => {
        const filtered = mockCases.filter(
            (caseItem) =>
                caseItem.title.toLowerCase().includes(searchTerm.toLowerCase()) ||
                caseItem.description.toLowerCase().includes(searchTerm.toLowerCase()) ||
                caseItem.type.toLowerCase().includes(searchTerm.toLowerCase()),
        )
        setFilteredCases(filtered)
    }, [searchTerm])

    return (
        <div className="min-h-screen bg-background">
            <main className="container mx-auto px-4 py-8">
                {/* <div className="text-center mb-12">
                    <h1 className="text-4xl md:text-6xl font-bold mb-6 bg-gradient-to-r from-primary to-accent bg-clip-text text-transparent">
                        AI Courtroom Simulation
                    </h1>
                    <p className="text-lg md:text-xl text-muted-foreground mb-8 max-w-3xl mx-auto leading-relaxed">
                        Experience realistic legal proceedings with AI-powered judges and attorneys. Practice your legal skills in a
                        safe, educational environment.
                    </p>
                </div> */}

                <div className="grid grid-cols-2 md:grid-cols-4 gap-4 md:gap-6 mb-12">
                    <div className="bg-card p-4 md:p-6 rounded-xl border shadow-sm hover:shadow-md transition-shadow">
                        <FileText className="h-6 w-6 md:h-8 md:w-8 text-primary mx-auto mb-2" />
                        <div className="text-xl md:text-2xl font-bold text-card-foreground">150+</div>
                        <div className="text-xs md:text-sm text-muted-foreground">Active Cases</div>
                    </div>
                    <div className="bg-card p-4 md:p-6 rounded-xl border shadow-sm hover:shadow-md transition-shadow">
                        <Users className="h-6 w-6 md:h-8 md:w-8 text-primary mx-auto mb-2" />
                        <div className="text-xl md:text-2xl font-bold text-card-foreground">5,000+</div>
                        <div className="text-xs md:text-sm text-muted-foreground">Users</div>
                    </div>
                    <div className="bg-card p-4 md:p-6 rounded-xl border shadow-sm hover:shadow-md transition-shadow">
                        <MessageSquare className="h-6 w-6 md:h-8 md:w-8 text-primary mx-auto mb-2" />
                        <div className="text-xl md:text-2xl font-bold text-card-foreground">25,000+</div>
                        <div className="text-xs md:text-sm text-muted-foreground">Simulations</div>
                    </div>
                    <div className="bg-card p-4 md:p-6 rounded-xl border shadow-sm hover:shadow-md transition-shadow">
                        <TrendingUp className="h-6 w-6 md:h-8 md:w-8 text-primary mx-auto mb-2" />
                        <div className="text-xl md:text-2xl font-bold text-card-foreground">98%</div>
                        <div className="text-xs md:text-sm text-muted-foreground">Success Rate</div>
                    </div>
                </div>

                <div className="relative mb-8 max-w-2xl mx-auto">
                    <Search className="absolute left-3 top-1/2 transform -translate-y-1/2 text-muted-foreground h-5 w-5" />
                    <input
                        type="text"
                        placeholder="Search cases by title, description, or type..."
                        value={searchTerm}
                        onChange={(e) => setSearchTerm(e.target.value)}
                        className="w-full pl-10 pr-4 py-3 border border-input rounded-xl bg-background text-foreground placeholder:text-muted-foreground focus:outline-none focus:ring-2 focus:ring-primary focus:border-transparent shadow-sm"
                    />
                </div>

                <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6 lg:gap-8">
                    {filteredCases.length > 0 ? (
                        filteredCases.map((caseItem) => <CaseCard key={caseItem.id} caseData={caseItem} />)
                    ) : (
                        <div className="col-span-full text-center py-16">
                            <FileText className="h-16 w-16 text-muted-foreground mx-auto mb-4" />
                            <h3 className="text-xl font-semibold text-foreground mb-2">No cases found</h3>
                            <p className="text-muted-foreground">
                                {searchTerm ? "Try adjusting your search terms" : "No cases available at the moment"}
                            </p>
                        </div>
                    )}
                </div>
            </main>
        </div>
    )
}
