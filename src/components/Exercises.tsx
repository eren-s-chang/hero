import { ArrowRight } from "lucide-react";

const exercises = [
  {
    name: "Squat",
    description: "Knee tracking, depth, back angle, and weight distribution analysis.",
    tag: "Lower Body",
  },
  {
    name: "Deadlift",
    description: "Spinal alignment, hip hinge mechanics, and lockout form detection.",
    tag: "Posterior Chain",
  },
  {
    name: "Push-Up",
    description: "Elbow flare, core sag, and range of motion evaluation.",
    tag: "Upper Body",
  },
];

const Exercises = () => {
  return (
    <section className="py-24 px-6 border-t border-border">
      <div className="max-w-6xl mx-auto">
        <p className="text-primary font-medium tracking-widest uppercase text-sm mb-3">Supported Exercises</p>
        <h2 className="font-heading text-4xl md:text-5xl mb-16">
          Precision analysis for<br />key movements.
        </h2>
        <div className="space-y-4">
          {exercises.map((ex) => (
            <div
              key={ex.name}
              className="flex flex-col md:flex-row md:items-center justify-between bg-card rounded-xl p-6 md:p-8 border border-border hover:border-primary/40 transition-colors group cursor-pointer"
            >
              <div className="flex items-center gap-6 mb-4 md:mb-0">
                <h3 className="font-heading text-3xl md:text-4xl">{ex.name}</h3>
                <span className="text-xs tracking-widest uppercase text-primary bg-primary/10 px-3 py-1 rounded-full">
                  {ex.tag}
                </span>
              </div>
              <div className="flex items-center gap-6">
                <p className="text-muted-foreground text-sm max-w-xs">{ex.description}</p>
                <ArrowRight className="w-5 h-5 text-muted-foreground group-hover:text-primary transition-colors shrink-0" />
              </div>
            </div>
          ))}
        </div>
      </div>
    </section>
  );
};

export default Exercises;
