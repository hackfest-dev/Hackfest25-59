import React from "react"
import { cn } from "@/lib/utils"

interface AIAnimationProps {
    className?: string
    isTalking?: boolean
    isThinking?: boolean
}

export function AIAnimation({ className, isTalking = false, isThinking = false }: AIAnimationProps) {
    return (
        <div
            className={cn(
                "relative w-full max-w-md aspect-video bg-gradient-to-br from-slate-900 to-blue-900 rounded-lg overflow-hidden flex items-center justify-center",
                className
            )}
        >
            {/* Holographic Grid Background */}
            <div className="absolute inset-0 opacity-30">
                <div className="absolute inset-0 grid grid-cols-12 grid-rows-8 gap-px">
                    {[...Array(96)].map((_, i) => (
                        <div
                            key={i}
                            className="bg-blue-500/10 rounded-sm"
                            style={{
                                animation: `gridPulse ${2 + Math.random() * 2}s infinite ease-in-out`,
                                animationDelay: `${Math.random() * 2}s`,
                            }}
                        />
                    ))}
                </div>
            </div>

            {/* Main Circle Container */}
            <div className="relative w-48 h-48 flex items-center justify-center">
                {/* Outer Glow Ring */}
                <div
                    className={cn(
                        "absolute inset-0 rounded-full border-4",
                        isTalking
                            ? "border-blue-400/50 animate-pulse"
                            : isThinking
                            ? "border-purple-400/50 animate-pulse"
                            : "border-blue-500/20"
                    )}
                />

                {/* Animated Rings */}
                <div
                    className={cn(
                        "absolute inset-0 rounded-full border-4",
                        isTalking
                            ? "border-blue-400/30 animate-spin-slow"
                            : isThinking
                            ? "border-purple-400/30 animate-spin-slow-reverse"
                            : "border-blue-500/10 animate-spin-slow"
                    )}
                />

                {/* Inner Circle */}
                <div
                    className={cn(
                        "absolute inset-4 rounded-full",
                        isTalking
                            ? "bg-gradient-to-br from-blue-400 to-blue-600 animate-pulse"
                            : isThinking
                            ? "bg-gradient-to-br from-purple-400 to-blue-500 animate-pulse"
                            : "bg-gradient-to-br from-blue-500 to-blue-700"
                    )}
                />

                {/* Talking Animation - Wave Effect */}
                {isTalking && (
                    <div className="absolute inset-0 flex items-center justify-center">
                        {[...Array(5)].map((_, i) => (
                            <div
                                key={i}
                                className="absolute w-full h-full rounded-full border-2 border-blue-400/30"
                                style={{
                                    animation: `wavePulse 2s infinite ease-in-out`,
                                    animationDelay: `${i * 0.2}s`,
                                    transform: `scale(${1 + i * 0.2})`,
                                }}
                            />
                        ))}
                    </div>
                )}

                {/* Thinking Animation Elements */}
                {isThinking && (
                    <>
                        {/* Central Pulsing Circle */}
                        <div className="absolute inset-8 rounded-full bg-purple-400/30 animate-pulse" />
                        <div
                            className="absolute inset-12 rounded-full bg-purple-400/20 animate-pulse"
                            style={{ animationDelay: "0.5s" }}
                        />

                        {/* Thinking Wave Pattern */}
                        <div className="absolute inset-0 flex items-center justify-center">
                            {[...Array(3)].map((_, i) => (
                                <div
                                    key={i}
                                    className="absolute w-full h-full rounded-full"
                                    style={{
                                        animation: `thinkingWave 2s infinite ease-in-out`,
                                        animationDelay: `${i * 0.3}s`,
                                        transform: `scale(${1 + i * 0.3})`,
                                        background: `radial-gradient(circle, rgba(147, 51, 234, ${
                                            0.3 - i * 0.05
                                        }) 0%, rgba(147, 51, 234, 0) 70%)`,
                                    }}
                                />
                            ))}
                        </div>

                        {/* Thinking Dots */}
                        <div className="absolute inset-0 flex items-center justify-center">
                            {[...Array(3)].map((_, i) => (
                                <div
                                    key={i}
                                    className="absolute w-3 h-3 rounded-full bg-purple-400"
                                    style={{
                                        transform: `translateX(${(i - 1) * 24}px)`,
                                        animation: `thinkingDot 1.5s infinite ease-in-out`,
                                        animationDelay: `${i * 0.2}s`,
                                    }}
                                />
                            ))}
                        </div>

                        {/* Additional Thinking Elements */}
                        <div className="absolute inset-0 flex items-center justify-center">
                            {[...Array(8)].map((_, i) => (
                                <div
                                    key={i}
                                    className="absolute w-2 h-2 rounded-full bg-purple-400/70"
                                    style={{
                                        transform: `rotate(${i * 45}deg) translateY(-30px)`,
                                        animation: `thinkingPulse 2s infinite ease-in-out`,
                                        animationDelay: `${i * 0.15}s`,
                                    }}
                                />
                            ))}
                        </div>
                    </>
                )}

                {/* Floating Particles */}
                <div className="absolute inset-0 overflow-hidden">
                    {[...Array(20)].map((_, i) => (
                        <div
                            key={i}
                            className={cn(
                                "absolute w-1 h-1 rounded-full",
                                isTalking ? "bg-blue-400" : isThinking ? "bg-purple-400" : "bg-blue-500"
                            )}
                            style={{
                                left: `${Math.random() * 100}%`,
                                top: `${Math.random() * 100}%`,
                                animation: `particleFloat ${3 + Math.random() * 2}s infinite ease-in-out`,
                                animationDelay: `${Math.random() * 2}s`,
                            }}
                        />
                    ))}
                </div>
            </div>

            {/* Ambient Glow */}
            <div
                className={cn(
                    "absolute inset-0 bg-gradient-to-br",
                    isTalking
                        ? "from-blue-500/10 to-purple-500/10 animate-pulse"
                        : isThinking
                        ? "from-purple-500/20 to-blue-500/20 animate-pulse"
                        : "from-blue-500/5 to-purple-500/5"
                )}
            />
        </div>
    )
}
