"use client"
import { useState } from "react"
import { Heart, MessageCircle, Eye, Play, Download, User } from "lucide-react"

export default function CaseCard({ caseData }) {
    const [liked, setLiked] = useState(false)
    const [showComments, setShowComments] = useState(false)
    const [comments, setComments] = useState([
        { id: 1, user: "John Doe", text: "Great case study!", time: "2 hours ago" },
        { id: 2, user: "Jane Smith", text: "Very informative evidence.", time: "1 day ago" },
    ])
    const [newComment, setNewComment] = useState("")

    const handleLike = () => {
        setLiked(!liked)
        // TODO: Backend - Update like status in database
    }

    const handleComment = () => {
        setShowComments(!showComments)
    }

    const handleAddComment = () => {
        if (newComment.trim()) {
            const comment = {
                id: comments.length + 1,
                user: "Current User", // TODO: Get from auth context
                text: newComment,
                time: "Just now",
            }
            setComments([...comments, comment])
            setNewComment("")
            // TODO: Backend - Add comment to database
        }
    }

    const handleKeyPress = (e) => {
        if (e.key === "Enter") {
            handleAddComment()
        }
    }

    const handleDownload = () => {
        console.log("Downloading case details...")
        // TODO: Backend - Generate and download case document
    }

    return (
        <div className="bg-card rounded-xl shadow-lg hover:shadow-xl transition-all duration-300 p-6 border border-border h-full flex flex-col group hover:scale-[1.02]">
            {/* Publisher */}
            <div className="flex items-center justify-between mb-4">
                <div className="flex items-center gap-3">
                    <div className="w-10 h-10 bg-gradient-to-br from-primary to-accent rounded-full flex items-center justify-center">
                        <User size={16} className="text-primary-foreground" />
                    </div>
                    <div>
                        <p className="text-sm font-medium text-card-foreground">{caseData.publishedBy || "Anonymous User"}</p>
                        <p className="text-xs text-muted-foreground">{caseData.publisherRole || "Legal Professional"}</p>
                    </div>
                </div>
                <div className="text-xs text-muted-foreground">
                    {new Date(caseData.created_at).toLocaleDateString()}
                </div>
            </div>

            {/* Case Content */}
            <div className="flex-1">
                <div className="mb-4">
                    <h3 className="text-lg sm:text-xl font-bold text-card-foreground mb-3 line-clamp-2 leading-tight group-hover:text-primary transition-colors">
                        {caseData.title}
                    </h3>

                    <span className="inline-block px-4 py-2 text-xs sm:text-sm font-semibold bg-gradient-to-r from-primary to-accent text-primary-foreground rounded-full mb-3 shadow-md">
                        {caseData.type}
                    </span>

                    <p className="text-sm sm:text-base text-muted-foreground line-clamp-3 leading-relaxed">
                        {caseData.description}
                    </p>
                </div>

                {/* Buttons */}
                <div className="grid grid-cols-1 sm:grid-cols-3 gap-2 mb-4">
                    <button
                        onClick={() => (window.location.href = `/simulation/${caseData.id}`)}
                        className="flex items-center justify-center gap-2 px-4 py-3 bg-gradient-to-r from-primary to-accent hover:from-primary/90 hover:to-accent/90 text-primary-foreground rounded-lg transition-all duration-200 text-sm font-semibold shadow-md hover:shadow-lg transform hover:-translate-y-0.5"
                    >
                        <Play size={16} />
                        <span>Start Simulation</span>
                    </button>
                    <button className="flex items-center justify-center gap-2 px-4 py-3 bg-secondary hover:bg-secondary/90 text-secondary-foreground rounded-lg transition-all duration-200 text-sm font-semibold shadow-md hover:shadow-lg transform hover:-translate-y-0.5">
                        <Eye size={16} />
                        <span>Analysis</span>
                    </button>
                    <button
                        onClick={handleDownload}
                        className="flex items-center justify-center gap-2 px-4 py-3 bg-accent hover:bg-accent/90 text-accent-foreground rounded-lg transition-all duration-200 text-sm font-semibold shadow-md hover:shadow-lg transform hover:-translate-y-0.5"
                    >
                        <Download size={16} />
                        <span>Download</span>
                    </button>
                </div>
            </div>

            {/* Footer (likes + comments + difficulty) */}
            <div className="flex items-center justify-between pt-4 border-t border-border mt-auto">
                <div className="flex items-center gap-4">
                    <button
                        onClick={handleLike}
                        className={`flex items-center gap-2 px-3 py-2 rounded-lg transition-all duration-200 text-sm font-medium ${liked
                                ? "text-red-600 bg-red-50 dark:bg-red-900/20 shadow-md"
                                : "text-muted-foreground hover:bg-muted hover:shadow-md"
                            }`}
                    >
                        <Heart size={16} fill={liked ? "currentColor" : "none"} />
                        <span>{caseData.likes + (liked ? 1 : 0)}</span>
                    </button>
                    <button
                        onClick={handleComment}
                        className="flex items-center gap-2 px-3 py-2 text-muted-foreground hover:bg-muted rounded-lg transition-all duration-200 text-sm font-medium hover:shadow-md"
                    >
                        <MessageCircle size={16} />
                        <span>{comments.length}</span>
                    </button>
                </div>
                <div className="flex items-center gap-2">
                    <span className="text-xs text-muted-foreground">Difficulty:</span>
                    <div className="flex gap-1">
                        {[...Array(5)].map((_, i) => (
                            <div
                                key={i}
                                className={`w-2 h-2 rounded-full ${i < (caseData.difficulty || 3)
                                        ? "bg-gradient-to-r from-primary to-accent"
                                        : "bg-muted"
                                    }`}
                            />
                        ))}
                    </div>
                </div>
            </div>

            {/* Comments */}
            {showComments && (
                <div className="mt-4 pt-4 border-t border-border bg-muted/50 rounded-lg p-4 -mx-2">
                    <div className="max-h-40 overflow-y-auto mb-3 space-y-3">
                        {comments.map((comment) => (
                            <div key={comment.id} className="p-3 bg-card rounded-lg shadow-sm border border-border">
                                <div className="flex justify-between items-start mb-2">
                                    <span className="font-semibold text-sm text-card-foreground">{comment.user}</span>
                                    <span className="text-xs text-muted-foreground">{comment.time}</span>
                                </div>
                                <p className="text-sm text-muted-foreground leading-relaxed">{comment.text}</p>
                            </div>
                        ))}
                    </div>
                    <div className="flex gap-2">
                        <input
                            type="text"
                            value={newComment}
                            onChange={(e) => setNewComment(e.target.value)}
                            onKeyPress={handleKeyPress}
                            placeholder="Add a comment..."
                            className="flex-1 px-4 py-2 text-sm border border-input rounded-lg bg-background text-foreground placeholder-muted-foreground focus:outline-none focus:ring-2 focus:ring-primary focus:border-transparent shadow-sm"
                        />
                        <button
                            onClick={handleAddComment}
                            className="px-4 py-2 bg-gradient-to-r from-primary to-accent hover:from-primary/90 hover:to-accent/90 text-primary-foreground rounded-lg transition-all duration-200 text-sm font-semibold shadow-md hover:shadow-lg"
                        >
                            Send
                        </button>
                    </div>
                </div>
            )}
        </div>
    )
}

