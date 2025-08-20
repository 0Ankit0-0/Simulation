"use client"
import { useState, useEffect } from "react"
import Navbar from "../../components/navbar"
import { ThemeProvider } from "../../components/theme-provider"
import { Card, CardContent, CardHeader, CardTitle } from "../../components/ui/card"
import { Badge } from "../ui/badge"
import { Button } from "../ui/button"
import { Avatar, AvatarFallback, AvatarImage } from "../ui/avatar"
import {
    Mail,
    Phone,
    MapPin,
    Building,
    GraduationCap,
    BookOpen,
    Edit,
    Save,
    X,
    Eye,
    EyeOff,
    Heart,
    MessageCircle,
} from "lucide-react"
import axios from "axios"

// Mock user data - in real app this would come from authentication/database
const mockUser = {
    id: 1,
    name: "Sarah Johnson",
    email: "sarah.johnson@lawschool.edu",
    phone: "+1 (555) 123-4567",
    location: "New York, NY",
    role: "student", // student, law_firm, professor
    avatar: "/abstract-profile.png",

    // Student specific fields
    collegeName: "Harvard Law School",
    year: "3rd Year",
    gpa: "3.8",
    specialization: "Criminal Law",

    // Law firm specific fields
    firmName: "Johnson & Associates",
    position: "Senior Partner",
    barNumber: "NY12345",
    practiceAreas: ["Criminal Defense", "Corporate Law", "Family Law"],
    yearsExperience: 15,

    // Professor specific fields
    university: "Columbia Law School",
    department: "Criminal Justice",
    tenure: "Tenured Professor",
    courses: ["Constitutional Law", "Criminal Procedure", "Evidence"],
    publications: 23,
}

// Mock published cases data
const mockPublishedCases = [
    {
        id: 1,
        title: "Landmark Constitutional Rights Case",
        description:
            "A comprehensive case study examining the balance between individual rights and state security measures.",
        type: "Constitutional Law",
        likes: 89,
        comments: 34,
        isPublic: true,
        createdAt: "2024-01-15",
    },
    {
        id: 2,
        title: "Complex Corporate Merger Analysis",
        description: "Detailed examination of antitrust implications in a major corporate acquisition scenario.",
        type: "Corporate Law",
        likes: 67,
        comments: 28,
        isPublic: false,
        createdAt: "2024-02-03",
    },
    {
        id: 3,
        title: "Environmental Law Precedent Study",
        description: "Analysis of environmental protection laws and their enforcement in industrial contexts.",
        type: "Environmental Law",
        likes: 45,
        comments: 19,
        isPublic: true,
        createdAt: "2024-02-20",
    },
]

export default function ProfilePage({ userId }) {
    const [user, setUser] = useState(mockUser)
    const [isEditing, setIsEditing] = useState(false)
    const [editedUser, setEditedUser] = useState(mockUser)
    const [publishedCases, setPublishedCases] = useState([])
    const [showComments, setShowComments] = useState({})

    useEffect(() => {
        const fetchUserProfile = async () => {
            try {
                const response = await axios.get(`/api/users/${userId}`)
                setUser(response.data)
                setEditedUser(response.data)
            } catch (error) {
                console.error("Error fetching user profile:", error)
            }
        }

        const fetchPublishedCases = async () => {
            try {
                const response = await axios.get(`/api/users/${userId}/published-cases`)
                setPublishedCases(Array.isArray(response.data) ? response.data : mockPublishedCases)
            } catch (error) {
                console.error("Error fetching published cases:", error)
                setPublishedCases(mockPublishedCases)
            }
        }

        setPublishedCases(mockPublishedCases)

        fetchUserProfile()
        fetchPublishedCases()
    }, [userId])

    const handleEdit = () => {
        setIsEditing(true)
        setEditedUser(user)
    }

    const handleSave = async () => {
        try {
            const response = await axios.put(`/api/users/${userId}`, editedUser)
            setUser(response.data)
            setIsEditing(false)
        } catch (error) {
            console.error("Error updating user profile:", error)
        }
    }

    const handleCancel = () => {
        setEditedUser(user)
        setIsEditing(false)
    }

    const handleInputChange = (field, value) => {
        setEditedUser((prev) => ({ ...prev, [field]: value }))
    }

    const toggleCasePrivacy = async (caseId) => {
        try {
            const response = await axios.put(`/api/users/${userId}/cases/${caseId}/privacy`)
            setPublishedCases((prev) =>
                prev.map((caseItem) => (caseItem.id === caseId ? { ...caseItem, isPublic: response.data.isPublic } : caseItem)),
            )
        } catch (error) {
            console.error("Error toggling case privacy:", error)
        }
    }

    const toggleComments = (caseId) => {
        setShowComments((prev) => ({
            ...prev,
            [caseId]: !prev[caseId],
        }))
    }

    const renderRoleSpecificContent = () => {
        const currentUser = isEditing ? editedUser : user

        switch (currentUser.role) {
            case "student":
                return (
                    <Card>
                        <CardHeader>
                            <CardTitle className="flex items-center gap-2">
                                <GraduationCap className="h-5 w-5" />
                                Academic Information
                            </CardTitle>
                        </CardHeader>
                        <CardContent className="space-y-4">
                            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                                <div>
                                    <label className="text-sm font-medium text-muted-foreground">College/University</label>
                                    {isEditing ? (
                                        <input
                                            type="text"
                                            value={currentUser.collegeName}
                                            onChange={(e) => handleInputChange("collegeName", e.target.value)}
                                            className="w-full mt-1 px-3 py-2 border rounded-md bg-background"
                                        />
                                    ) : (
                                        <p className="text-foreground font-medium">{currentUser.collegeName}</p>
                                    )}
                                </div>
                                <div>
                                    <label className="text-sm font-medium text-muted-foreground">Year</label>
                                    {isEditing ? (
                                        <input
                                            type="text"
                                            value={currentUser.year}
                                            onChange={(e) => handleInputChange("year", e.target.value)}
                                            className="w-full mt-1 px-3 py-2 border rounded-md bg-background"
                                        />
                                    ) : (
                                        <p className="text-foreground font-medium">{currentUser.year}</p>
                                    )}
                                </div>
                                <div>
                                    <label className="text-sm font-medium text-muted-foreground">GPA</label>
                                    {isEditing ? (
                                        <input
                                            type="text"
                                            value={currentUser.gpa}
                                            onChange={(e) => handleInputChange("gpa", e.target.value)}
                                            className="w-full mt-1 px-3 py-2 border rounded-md bg-background"
                                        />
                                    ) : (
                                        <p className="text-foreground font-medium">{currentUser.gpa}</p>
                                    )}
                                </div>
                                <div>
                                    <label className="text-sm font-medium text-muted-foreground">Specialization</label>
                                    {isEditing ? (
                                        <input
                                            type="text"
                                            value={currentUser.specialization}
                                            onChange={(e) => handleInputChange("specialization", e.target.value)}
                                            className="w-full mt-1 px-3 py-2 border rounded-md bg-background"
                                        />
                                    ) : (
                                        <p className="text-foreground font-medium">{currentUser.specialization}</p>
                                    )}
                                </div>
                            </div>
                        </CardContent>
                    </Card>
                )

            case "law_firm":
                return (
                    <Card>
                        <CardHeader>
                            <CardTitle className="flex items-center gap-2">
                                <Building className="h-5 w-5" />
                                Professional Information
                            </CardTitle>
                        </CardHeader>
                        <CardContent className="space-y-4">
                            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                                <div>
                                    <label className="text-sm font-medium text-muted-foreground">Law Firm</label>
                                    {isEditing ? (
                                        <input
                                            type="text"
                                            value={currentUser.firmName}
                                            onChange={(e) => handleInputChange("firmName", e.target.value)}
                                            className="w-full mt-1 px-3 py-2 border rounded-md bg-background"
                                        />
                                    ) : (
                                        <p className="text-foreground font-medium">{currentUser.firmName}</p>
                                    )}
                                </div>
                                <div>
                                    <label className="text-sm font-medium text-muted-foreground">Position</label>
                                    {isEditing ? (
                                        <input
                                            type="text"
                                            value={currentUser.position}
                                            onChange={(e) => handleInputChange("position", e.target.value)}
                                            className="w-full mt-1 px-3 py-2 border rounded-md bg-background"
                                        />
                                    ) : (
                                        <p className="text-foreground font-medium">{currentUser.position}</p>
                                    )}
                                </div>
                                <div>
                                    <label className="text-sm font-medium text-muted-foreground">Bar Number</label>
                                    {isEditing ? (
                                        <input
                                            type="text"
                                            value={currentUser.barNumber}
                                            onChange={(e) => handleInputChange("barNumber", e.target.value)}
                                            className="w-full mt-1 px-3 py-2 border rounded-md bg-background"
                                        />
                                    ) : (
                                        <p className="text-foreground font-medium">{currentUser.barNumber}</p>
                                    )}
                                </div>
                                <div>
                                    <label className="text-sm font-medium text-muted-foreground">Years of Experience</label>
                                    {isEditing ? (
                                        <input
                                            type="number"
                                            value={currentUser.yearsExperience}
                                            onChange={(e) => handleInputChange("yearsExperience", e.target.value)}
                                            className="w-full mt-1 px-3 py-2 border rounded-md bg-background"
                                        />
                                    ) : (
                                        <p className="text-foreground font-medium">{currentUser.yearsExperience} years</p>
                                    )}
                                </div>
                            </div>
                            <div>
                                <label className="text-sm font-medium text-muted-foreground">Practice Areas</label>
                                <div className="flex flex-wrap gap-2 mt-2">
                                    {currentUser.practiceAreas &&
                                        Array.isArray(currentUser.practiceAreas) &&
                                        currentUser.practiceAreas.map((area, index) => (
                                            <Badge key={index} variant="secondary">
                                                {area}
                                            </Badge>
                                        ))}
                                </div>
                            </div>
                        </CardContent>
                    </Card>
                )

            case "professor":
                return (
                    <Card>
                        <CardHeader>
                            <CardTitle className="flex items-center gap-2">
                                <BookOpen className="h-5 w-5" />
                                Academic Information
                            </CardTitle>
                        </CardHeader>
                        <CardContent className="space-y-4">
                            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                                <div>
                                    <label className="text-sm font-medium text-muted-foreground">University</label>
                                    {isEditing ? (
                                        <input
                                            type="text"
                                            value={currentUser.university}
                                            onChange={(e) => handleInputChange("university", e.target.value)}
                                            className="w-full mt-1 px-3 py-2 border rounded-md bg-background"
                                        />
                                    ) : (
                                        <p className="text-foreground font-medium">{currentUser.university}</p>
                                    )}
                                </div>
                                <div>
                                    <label className="text-sm font-medium text-muted-foreground">Department</label>
                                    {isEditing ? (
                                        <input
                                            type="text"
                                            value={currentUser.department}
                                            onChange={(e) => handleInputChange("department", e.target.value)}
                                            className="w-full mt-1 px-3 py-2 border rounded-md bg-background"
                                        />
                                    ) : (
                                        <p className="text-foreground font-medium">{currentUser.department}</p>
                                    )}
                                </div>
                                <div>
                                    <label className="text-sm font-medium text-muted-foreground">Status</label>
                                    {isEditing ? (
                                        <input
                                            type="text"
                                            value={currentUser.tenure}
                                            onChange={(e) => handleInputChange("tenure", e.target.value)}
                                            className="w-full mt-1 px-3 py-2 border rounded-md bg-background"
                                        />
                                    ) : (
                                        <p className="text-foreground font-medium">{currentUser.tenure}</p>
                                    )}
                                </div>
                                <div>
                                    <label className="text-sm font-medium text-muted-foreground">Publications</label>
                                    {isEditing ? (
                                        <input
                                            type="number"
                                            value={currentUser.publications}
                                            onChange={(e) => handleInputChange("publications", e.target.value)}
                                            className="w-full mt-1 px-3 py-2 border rounded-md bg-background"
                                        />
                                    ) : (
                                        <p className="text-foreground font-medium">{currentUser.publications} publications</p>
                                    )}
                                </div>
                            </div>
                            <div>
                                <label className="text-sm font-medium text-muted-foreground">Courses Teaching</label>
                                <div className="flex flex-wrap gap-2 mt-2">
                                    {currentUser.courses &&
                                        Array.isArray(currentUser.courses) &&
                                        currentUser.courses.map((course, index) => (
                                            <Badge key={index} variant="secondary">
                                                {course}
                                            </Badge>
                                        ))}
                                </div>
                            </div>
                        </CardContent>
                    </Card>
                )

            default:
                return null
        }
    }

    const currentUser = isEditing ? editedUser : user

    return (
        <ThemeProvider attribute="class" defaultTheme="system" enableSystem disableTransitionOnChange>
            <div className="min-h-screen bg-background">
                <Navbar />

                <main className="container mx-auto px-4 py-8">
                    <div className="max-w-4xl mx-auto space-y-6">
                        {/* Profile Header */}
                        <Card>
                            <CardContent className="pt-6">
                                <div className="flex flex-col md:flex-row items-start md:items-center gap-6">
                                    <Avatar className="h-24 w-24">
                                        <AvatarImage src={currentUser.avatar || "/placeholder.svg"} alt={currentUser.name} />
                                        <AvatarFallback className="text-2xl">
                                            {currentUser.name &&
                                                currentUser.name
                                                    .split(" ")
                                                    .map((n) => n[0])
                                                    .join("")}
                                        </AvatarFallback>
                                    </Avatar>

                                    <div className="flex-1 space-y-2">
                                        <div className="flex items-center justify-between">
                                            <div>
                                                {isEditing ? (
                                                    <input
                                                        type="text"
                                                        value={currentUser.name}
                                                        onChange={(e) => handleInputChange("name", e.target.value)}
                                                        className="text-2xl font-bold bg-background border rounded px-2 py-1"
                                                    />
                                                ) : (
                                                    <h1 className="text-2xl font-bold text-foreground">{currentUser.name}</h1>
                                                )}
                                                <Badge variant="outline" className="mt-1 capitalize">
                                                    {currentUser.role.replace("_", " ")}
                                                </Badge>
                                            </div>

                                            <div className="flex gap-2">
                                                {isEditing ? (
                                                    <>
                                                        <Button onClick={handleSave} size="sm">
                                                            <Save className="h-4 w-4 mr-2" />
                                                            Save
                                                        </Button>
                                                        <Button onClick={handleCancel} variant="outline" size="sm">
                                                            <X className="h-4 w-4 mr-2" />
                                                            Cancel
                                                        </Button>
                                                    </>
                                                ) : (
                                                    <Button onClick={handleEdit} variant="outline" size="sm">
                                                        <Edit className="h-4 w-4 mr-2" />
                                                        Edit Profile
                                                    </Button>
                                                )}
                                            </div>
                                        </div>

                                        <div className="grid grid-cols-1 md:grid-cols-3 gap-4 text-sm">
                                            <div className="flex items-center gap-2 text-muted-foreground">
                                                <Mail className="h-4 w-4" />
                                                {isEditing ? (
                                                    <input
                                                        type="email"
                                                        value={currentUser.email}
                                                        onChange={(e) => handleInputChange("email", e.target.value)}
                                                        className="bg-background border rounded px-2 py-1 flex-1"
                                                    />
                                                ) : (
                                                    <span>{currentUser.email}</span>
                                                )}
                                            </div>
                                            <div className="flex items-center gap-2 text-muted-foreground">
                                                <Phone className="h-4 w-4" />
                                                {isEditing ? (
                                                    <input
                                                        type="tel"
                                                        value={currentUser.phone}
                                                        onChange={(e) => handleInputChange("phone", e.target.value)}
                                                        className="bg-background border rounded px-2 py-1 flex-1"
                                                    />
                                                ) : (
                                                    <span>{currentUser.phone}</span>
                                                )}
                                            </div>
                                            <div className="flex items-center gap-2 text-muted-foreground">
                                                <MapPin className="h-4 w-4" />
                                                {isEditing ? (
                                                    <input
                                                        type="text"
                                                        value={currentUser.location}
                                                        onChange={(e) => handleInputChange("location", e.target.value)}
                                                        className="bg-background border rounded px-2 py-1 flex-1"
                                                    />
                                                ) : (
                                                    <span>{currentUser.location}</span>
                                                )}
                                            </div>
                                        </div>
                                    </div>
                                </div>
                            </CardContent>
                        </Card>

                        {/* Role-specific Information */}
                        {renderRoleSpecificContent()}

                        {/* Published Cases Section */}
                        <Card>
                            <CardHeader>
                                <CardTitle>Published Cases</CardTitle>
                            </CardHeader>
                            <CardContent>
                                <div className="space-y-4">
                                    {Array.isArray(publishedCases) && publishedCases.length > 0 ? (
                                        publishedCases.map((caseData) => (
                                            <div key={caseData.id} className="border rounded-lg p-4 space-y-3">
                                                <div className="flex items-start justify-between">
                                                    <div className="flex-1">
                                                        <div className="flex items-center gap-2 mb-2">
                                                            <h3 className="font-semibold">{caseData.title}</h3>
                                                            <Badge variant="outline">{caseData.type}</Badge>
                                                            <Badge variant={caseData.isPublic ? "default" : "secondary"}>
                                                                {caseData.isPublic ? "Public" : "Private"}
                                                            </Badge>
                                                        </div>
                                                        <p className="text-sm text-muted-foreground mb-2">{caseData.description}</p>
                                                        <p className="text-xs text-muted-foreground">Created: {caseData.createdAt}</p>
                                                    </div>
                                                    <Button
                                                        variant="outline"
                                                        size="sm"
                                                        onClick={() => toggleCasePrivacy(caseData.id)}
                                                        className="ml-4"
                                                    >
                                                        {caseData.isPublic ? <EyeOff className="h-4 w-4 mr-1" /> : <Eye className="h-4 w-4 mr-1" />}
                                                        {caseData.isPublic ? "Make Private" : "Make Public"}
                                                    </Button>
                                                </div>

                                                <div className="flex items-center justify-between pt-2 border-t">
                                                    <div className="flex items-center gap-4 text-sm text-muted-foreground">
                                                        <span className="flex items-center gap-1">
                                                            <Heart className="h-4 w-4" />
                                                            {caseData.likes}
                                                        </span>
                                                        <span className="flex items-center gap-1">
                                                            <MessageCircle className="h-4 w-4" />
                                                            {caseData.comments}
                                                        </span>
                                                    </div>
                                                    <Button variant="ghost" size="sm" onClick={() => toggleComments(caseData.id)}>
                                                        {showComments[caseData.id] ? "Hide Comments" : "View Comments"}
                                                    </Button>
                                                </div>

                                                {/* Comments Section */}
                                                {showComments[caseData.id] && (
                                                    <div className="mt-4 p-3 bg-muted rounded-lg">
                                                        <h4 className="font-medium mb-2">Recent Comments</h4>
                                                        <div className="space-y-2 text-sm">
                                                            <div className="flex gap-2">
                                                                <Avatar className="h-6 w-6">
                                                                    <AvatarFallback className="text-xs">JD</AvatarFallback>
                                                                </Avatar>
                                                                <div>
                                                                    <p className="font-medium">John Doe</p>
                                                                    <p className="text-muted-foreground">
                                                                        Excellent analysis of the constitutional implications!
                                                                    </p>
                                                                </div>
                                                            </div>
                                                            <div className="flex gap-2">
                                                                <Avatar className="h-6 w-6">
                                                                    <AvatarFallback className="text-xs">MS</AvatarFallback>
                                                                </Avatar>
                                                                <div>
                                                                    <p className="font-medium">Maria Smith</p>
                                                                    <p className="text-muted-foreground">
                                                                        This case study helped me understand the precedent better.
                                                                    </p>
                                                                </div>
                                                            </div>
                                                        </div>
                                                    </div>
                                                )}
                                            </div>
                                        ))
                                    ) : (
                                        <p className="text-muted-foreground text-center py-8">No published cases yet.</p>
                                    )}
                                </div>
                            </CardContent>
                        </Card>
                    </div>
                </main>
            </div>
        </ThemeProvider>
    )
}
