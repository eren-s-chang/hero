import { Upload, ScanSearch, ShieldCheck, ArrowRight } from "lucide-react";

const HowItWorks = () => {
  const steps = [
    {
      icon: Upload,
      title: "Upload Video",
      description: "Record or upload a video of your exercise — squat, deadlift, or push-up.",
    },
    {
      icon: ScanSearch,
      title: "Analyze Form",
      description: "Our AI analyzes every frame to detect alignment, depth, and movement patterns.",
    },
    {
      icon: ShieldCheck,
      title: "Get Safety Feedback",
      description: "Receive clear, actionable corrections to reduce injury risk and improve performance.",
    },
  ];

  return (
    <section className="py-24 px-6">
      <div className="max-w-6xl mx-auto">
        <p className="text-primary font-medium tracking-widest uppercase text-sm mb-3">How It Works</p>
        <h2 className="font-heading text-4xl md:text-5xl mb-16">
          Three steps to<br />safer movement.
        </h2>
        <div className="grid md:grid-cols-3 gap-8">
          {steps.map((step, i) => (
            <div
              key={step.title}
              className="bg-card rounded-xl p-8 border border-border hover:border-primary/40 transition-colors group"
            >
              <div className="flex items-center justify-between mb-6">
                <div className="w-12 h-12 rounded-lg bg-primary/10 flex items-center justify-center text-primary">
                  <step.icon className="w-6 h-6" />
                </div>
                <span className="text-muted-foreground font-heading text-2xl">0{i + 1}</span>
              </div>
              <h3 className="font-heading text-xl mb-3">{step.title}</h3>
              <p className="text-muted-foreground leading-relaxed text-sm">{step.description}</p>
            </div>
          ))}
        </div>
      </div>
    </section>
  );
};

export default HowItWorks;
