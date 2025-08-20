"use client"
import { Button } from "../ui/button"
import { Avatar, AvatarFallback, AvatarImage } from "../ui/avatar"
import {
    DropdownMenu,
    DropdownMenuContent,
    DropdownMenuItem,
    DropdownMenuTrigger,
    DropdownMenuSeparator,
} from "../ui/dropdown-menu"
import { Moon, Sun, Scale, LogOut, Plus, User, MessageCircle, Palette } from "lucide-react"
import { useTheme } from "next-themes"
import { Link, useNavigate } from "react-router-dom"
import { useState, useEffect } from "react"

function Navbar() {
    const { theme, setTheme } = useTheme()
    const [colorTheme, setColorTheme] = useState("blue")
    const [mounted, setMounted] = useState(false)
    const navigate = useNavigate()

    useEffect(() => {
        setMounted(true)
    }, [])

    useEffect(() => {
        if (mounted) {
            document.body.setAttribute("data-theme", colorTheme)
        }
    }, [colorTheme, mounted])

    const colorThemes = [
        { name: "Blue", value: "blue", colors: "from-blue-500 to-blue-600" },
        { name: "Purple", value: "purple", colors: "from-purple-500 to-purple-600" },
        { name: "Green", value: "green", colors: "from-green-500 to-green-600" },
        { name: "Red", value: "red", colors: "from-red-500 to-red-600" },
        { name: "Orange", value: "orange", colors: "from-orange-500 to-orange-600" },
        { name: "Pink", value: "pink", colors: "from-pink-500 to-pink-600" },
    ]

    const handleLogout = async () => {
        try {
            await fetch("/api/auth/logout", {
                method: "POST",
                headers: {
                    Authorization: `Bearer ${localStorage.getItem("auth_token")}`,
                    "Content-Type": "application/json",
                },
            })
            localStorage.removeItem("auth_token")
            localStorage.removeItem("user_data")
            navigate("/demopage")
        } catch (error) {
            console.error("Error logging out:", error)
        }
    }

    const handleChatbot = () => {
        navigate("/chatbot")
    }

    if (!mounted) return null

    return (
        <nav className="border-b bg-background/95 backdrop-blur supports-[backdrop-filter]:bg-background/60 shadow-sm">
            <div className="container mx-auto px-4">
                <div className="flex items-center justify-between h-16">
                    <Link to="/" className="flex items-center space-x-2 group">
                        <div className="p-2 bg-gradient-to-br from-primary to-accent rounded-lg shadow-md group-hover:shadow-lg transition-all duration-200">
                            <Scale className="h-6 w-6 text-white" />
                        </div>
                        <span className="text-xl font-bold bg-gradient-to-r from-primary to-accent bg-clip-text text-transparent">
                            AI Courtroom
                        </span>
                    </Link>

                    {/* Right Side */}
                    <div className="flex items-center space-x-4">
                        <Button
                            asChild
                            className="bg-gradient-to-r from-primary to-accent hover:from-primary/90 hover:to-accent/90 shadow-md hover:shadow-lg transition-all duration-200"
                        >
                            <Link to="/case" className="flex items-center space-x-2">
                                <Plus className="h-4 w-4" />
                                <span>Enter Case</span>
                            </Link>
                        </Button>

                        <Button variant="ghost" size="icon" onClick={handleChatbot}>
                            <MessageCircle className="h-4 w-4" />
                            <span className="sr-only">Open chatbot</span>
                        </Button>

                        {/* Theme Selector */}
                        <DropdownMenu>
                            <DropdownMenuTrigger asChild>
                                <Button variant="ghost" size="icon">
                                    <Palette className="h-4 w-4" />
                                    <span className="sr-only">Theme options</span>
                                </Button>
                            </DropdownMenuTrigger>
                            <DropdownMenuContent className="w-56" align="end">
                                <div className="p-2">
                                    <p className="text-sm font-medium mb-2">Theme Mode</p>
                                    <div className="flex gap-2 mb-3">
                                        <Button
                                            variant={theme === "light" ? "default" : "outline"}
                                            size="sm"
                                            onClick={() => setTheme("light")}
                                            className="flex-1"
                                        >
                                            <Sun className="h-4 w-4 mr-1" /> Light
                                        </Button>
                                        <Button
                                            variant={theme === "dark" ? "default" : "outline"}
                                            size="sm"
                                            onClick={() => setTheme("dark")}
                                            className="flex-1"
                                        >
                                            <Moon className="h-4 w-4 mr-1" /> Dark
                                        </Button>
                                    </div>
                                    <DropdownMenuSeparator />
                                    <p className="text-sm font-medium mb-2 mt-2">Color Theme</p>
                                    <div className="grid grid-cols-3 gap-2">
                                        {colorThemes.map((color) => (
                                            <button
                                                key={color.value}
                                                onClick={() => setColorTheme(color.value)}
                                                className={`p-2 rounded-lg border-2 transition-all ${colorTheme === color.value
                                                        ? "border-primary"
                                                        : "border-border hover:border-primary/50"
                                                    }`}
                                            >
                                                <div className={`w-full h-6 rounded bg-gradient-to-r ${color.colors} mb-1`} />
                                                <p className="text-xs font-medium">{color.name}</p>
                                            </button>
                                        ))}
                                    </div>
                                </div>
                            </DropdownMenuContent>
                        </DropdownMenu>

                        {/* Profile Dropdown */}
                        <DropdownMenu>
                            <DropdownMenuTrigger asChild>
                                <Button variant="ghost" className="relative h-8 w-8 rounded-full">
                                    <Avatar className="h-8 w-8">
                                        <AvatarImage src="/abstract-profile.png" alt="Profile" />
                                        <AvatarFallback>
                                            <User className="h-4 w-4" />
                                        </AvatarFallback>
                                    </Avatar>
                                </Button>
                            </DropdownMenuTrigger>
                            <DropdownMenuContent className="w-56" align="end">
                                <DropdownMenuItem asChild>
                                    <Link to="/profile" className="flex items-center">
                                        <User className="mr-2 h-4 w-4" />
                                        <span>Profile</span>
                                    </Link>
                                </DropdownMenuItem>
                                <DropdownMenuItem onClick={handleLogout}>
                                    <LogOut className="mr-2 h-4 w-4" />
                                    <span>Log out</span>
                                </DropdownMenuItem>
                            </DropdownMenuContent>
                        </DropdownMenu>
                    </div>
                </div>
            </div>
        </nav>
    )
}

export default Navbar
