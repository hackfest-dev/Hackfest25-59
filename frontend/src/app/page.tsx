"use client"

import { useState } from "react"
import { useRouter } from "next/navigation"
import { Input } from "@/components/ui/input"
import { Button } from "@/components/ui/button"
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from "@/components/ui/card"

export default function Home() {
    const [channelName, setChannelName] = useState<string>("")
    const [userName, setUserName] = useState<string>("")

    const router = useRouter()

    const onCallClick = (): void => {
        const query = new URLSearchParams({
            channelName: channelName || "",
            userName: userName || "",
        }).toString()
        router.push("/call?" + query)
    }

    return (
        <div className="flex flex-col items-center justify-center min-h-screen bg-gradient-to-br from-blue-50 via-purple-50 to-pink-50 dark:from-gray-900 dark:via-gray-800 dark:to-gray-900 p-6">
            <div className="w-full max-w-2xl">
                <Card className="bg-white/80 dark:bg-gray-900/80 backdrop-blur-sm shadow-2xl border-0">
                    <CardHeader className="space-y-1">
                        <CardTitle className="text-3xl font-bold text-center text-blue-700 dark:text-blue-400">
                            Welcome to Video Chat
                        </CardTitle>
                        <CardDescription className="text-center text-gray-600 dark:text-gray-400">
                            Connect with your peers through secure video calls
                        </CardDescription>
                    </CardHeader>
                    <CardContent className="space-y-6">
                        <div className="space-y-4">
                            <div className="space-y-2">
                                <label
                                    className="text-sm font-medium text-gray-700 dark:text-gray-200"
                                    htmlFor="channelName"
                                >
                                    Channel Name
                                </label>
                                <Input
                                    id="channelName"
                                    value={channelName}
                                    onChange={(e) => setChannelName(e.target.value)}
                                    placeholder="Enter a unique channel name"
                                    className="w-full bg-white/50 dark:bg-gray-800/50 border-gray-200 dark:border-gray-700"
                                />
                            </div>
                            <div className="space-y-2">
                                <label
                                    className="text-sm font-medium text-gray-700 dark:text-gray-200"
                                    htmlFor="userName"
                                >
                                    Your Name
                                </label>
                                <Input
                                    id="userName"
                                    value={userName}
                                    onChange={(e) => setUserName(e.target.value)}
                                    placeholder="Enter your display name"
                                    className="w-full bg-white/50 dark:bg-gray-800/50 border-gray-200 dark:border-gray-700"
                                />
                            </div>
                        </div>
                        <Button
                            className="w-full bg-gradient-to-r from-blue-600 to-purple-600 hover:from-blue-700 hover:to-purple-700 text-white shadow-lg transform transition-all duration-200 hover:scale-105"
                            onClick={onCallClick}
                        >
                            Start Call
                        </Button>
                    </CardContent>
                </Card>
            </div>
        </div>
    )
}
