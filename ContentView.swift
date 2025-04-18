import SwiftUI
import SceneKit

struct ContentView: View {
    @ObservedObject var pythonManager: PythonBackendManager
    @State private var isPressed = false
    @State private var ripplePhase = false
    @State private var isHovering = false
    @State private var rotationAngle: Double = 0
//    @State private var actionText = "Press to ask!"

    var body: some View {
        VStack(spacing: 20) {
            ZStack {
                // Main button
                Button(action: { 
                    pythonManager.currentActionText = "Asking.."
                    isPressed = true

                    // Send request to Python backend
                    pythonManager.sendSplashRequest()

                    // Reset the animation after a delay
                    DispatchQueue.main.asyncAfter(deadline: .now() + 2.5) {
                        isPressed = false
                    }
                }) {
                    ZStack {
                        Circle()
                            .fill(
                                AngularGradient(
                                    gradient: Gradient(colors: [
                                        Color(red: 77/255, green: 85/255, blue: 204/255),
                                        Color(red: 122/255, green: 115/255, blue: 209/255),
                                        Color(red: 77/255, green: 82/255, blue: 204/255)
                                    ]),
                                    center: .center,
//                                    startRadius: 10,
//                                    endRadius: 30
                                    angle: .degrees(rotationAngle)
//                                    endAngle: .degrees(rotationAngle+180)
//                                    startPoint: .topLeading,
//                                    endPoint: .bottomTrailing
                                )
                            )
                            .frame(width: 120, height: 120)
                            .shadow(color: .blue.opacity(0.3), radius: 10, x: 0, y: 5)
                            .opacity(0.7)
                            

                        // Water highlight
                        Circle()
                            .fill(Color.white.opacity(0.4))
                            .frame(width: 60, height: 30)
                            .offset(x: -15, y: -25)
                            .blur(radius: 8)

                        Text("Assistant")
                            .foregroundColor(.white)
                            .font(.system(size: 16, weight: .bold))
                            .shadow(color: .black.opacity(0.5), radius: 1)
                        
                        ParticleSystem(buttonSize: 300)
                            .allowsHitTesting(false)
                    }
                    .scaleEffect((isHovering && !isPressed) ? 1.03 : 1.0)
                    .scaleEffect(isPressed ? 0.9 : 1.0)
                    .animation(
                        .spring(response: 0.3, dampingFraction: 0.6), value: isPressed
                    )
                    .onHover { hovering in
                        withAnimation(.spring(response: 0.3, dampingFraction: 0.7)) {
                            isHovering = hovering
                        }
                    }
                    .onAppear {
                        withAnimation(.linear(duration: 10).repeatForever(autoreverses: false)) {
                            rotationAngle = 360
                        }
                    }
                }
                .buttonStyle(PlainButtonStyle())

                // Ripple effects
                ZStack {
                    // First ripple wave
                    ForEach(0..<5) { index in
                        WaterRipple(
                            isAnimating: isPressed,
                            delay: Double(index) * 0.1,
                            duration: 1.5,
                            scale: 1.0 + Double(index) * 0.4
                        )
                    }

                    // Central splash effect
                    Circle()
                        .stroke(Color.white.opacity(0.7), lineWidth: isPressed ? 2 : 0)
                        .frame(width: 70, height: 60)
                        .scaleEffect(isPressed ? 1.5 : 0.5)
                        .opacity(isPressed ? 0 : 1)
                        .animation(.easeOut(duration: 0.5), value: isPressed)
                }
            }
            .frame(height: 150)

            // Display response from Python backend

//            Text(pythonManager.lastResponse)
//                .font(.system(size: 12, design: .monospaced))
//                .foregroundColor(.green)
//                .frame(maxWidth: .infinity, alignment: .leading)
//                .padding(.horizontal)

//           if(pythonManager.lastResponse == "Recording cleaned up"){
//               actionText = "Asked!"
//           }
           Text(pythonManager.currentActionText)
                .font(.system(size: 14))
                .foregroundColor(.white)
        }
        .padding()
        .onAppear {
        // Start Python backend when app launches
        pythonManager.startPythonBackend()
    }
    }
}

// Water ripple component
struct WaterRipple: View {
    var isAnimating: Bool
    var delay: Double
    var duration: Double
    var scale: Double

    @State private var animating = false

    var body: some View {
        Circle()
            .stroke(
                LinearGradient(
                    gradient: Gradient(colors: [
                        Color.white.opacity(0.2),
                       Color.blue.opacity(0.4),
                        Color.blue.opacity(0.3)
                    ]),
                    startPoint: .top,
                    endPoint: .bottom
                ),
                lineWidth: animating ? 1 : 4
            )
            .frame(width: 140, height: 140)
            .scaleEffect(animating ? scale : 0.8)
            .opacity(animating ? 0 : 0.7)
            .blur(radius: animating ? 0.5 : 0)
            .animation(
                Animation
                    .easeOut(duration: duration)
                    .delay(delay)
                    .repeatCount(1, autoreverses: false),
                value: animating
            )
            .onChange(of: isAnimating) { newValue in
                if newValue {
                    DispatchQueue.main.asyncAfter(deadline: .now() + delay) {
                        animating = true

                        // Reset animation state after completion
                        DispatchQueue.main.asyncAfter(deadline: .now() + duration + 0.1) {
                            animating = false
                        }
                    }
                }
            }
    }
}

struct Particle: Identifiable {
    let id = UUID()
    var position: CGPoint
    var size: CGFloat
    var opacity: Double
    var speed: Double
    var angle: Double
    
    mutating func update() {
        position.x += CGFloat(cos(angle) * speed)
        position.y += CGFloat(sin(angle) * speed)
        opacity -= 0.01
        size -= 0.1
    }
    
    var isAlive: Bool {
        return opacity > 0 && size > 0
    }
}

struct ParticleSystem: View {
    @State private var particles: [Particle] = []
    @State private var timer: Timer?
    let buttonSize: CGFloat
    
    var body: some View {
        ZStack {
            ForEach(particles) { particle in
                Circle()
                    .fill(Color.white)
                    .frame(width: particle.size, height: particle.size)
                    .position(particle.position)
                    .opacity(particle.opacity)
                    .blur(radius: 0.5)
            }
        }
        .onAppear {
            timer = Timer.scheduledTimer(withTimeInterval: 0.05, repeats: true) { _ in
                updateParticles()
                
                if Int.random(in: 0...10) > 7 {
                    createParticle()
                }
            }
        }
        .onDisappear {
            timer?.invalidate()
        }
    }
    
    func createParticle() {
        let center = CGPoint(x: buttonSize / 2, y: buttonSize / 2)
        let radius = buttonSize / 2 - 2
        let angle = Double.random(in: 0..<2 * .pi)
        let x = center.x + cos(angle) * radius
        let y = center.y + sin(angle) * radius
        
        let particle = Particle(
            position: CGPoint(x: x, y: y),
            size: CGFloat.random(in: 1...3),
            opacity: Double.random(in: 0.3...0.8),
            speed: Double.random(in: 0.2...0.8),
            angle: angle
        )
        
        particles.append(particle)
    }
    
    func updateParticles() {
        for i in (0..<particles.count).reversed() {
            if i < particles.count {
                particles[i].update()
                
                if !particles[i].isAlive {
                    particles.remove(at: i)
                }
            }
        }
    }
}

#Preview {
    ContentView(pythonManager: EmptyPythonBackendManager())
        .frame(width: 300, height: 250)
//        .background(Color.gray)
}
