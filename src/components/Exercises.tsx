import { motion } from "framer-motion";
import { ArrowRight, Flame, Target, Dumbbell } from "lucide-react";

const exercises = [
  {
    name: "Squat",
    description: "Knee tracking, depth, back angle, and weight distribution analysis.",
    tag: "Lower Body",
    icon: Flame,
  },
  {
    name: "Deadlift",
    description: "Spinal alignment, hip hinge mechanics, and lockout form detection.",
    tag: "Posterior Chain",
    icon: Target,
  },
  {
    name: "Push-Up",
    description: "Elbow flare, core sag, and range of motion evaluation.",
    tag: "Upper Body",
    icon: Dumbbell,
  },
];

const Exercises = () => {
  const containerVariants = {
    hidden: { opacity: 0 },
    visible: {
      opacity: 1,
      transition: {
        staggerChildren: 0.15
      }
    }
  };

  const itemVariants = {
    hidden: { x: -100, opacity: 0 },
    visible: { 
      x: 0, 
      opacity: 1,
      transition: {
        type: "spring" as const,
        stiffness: 100,
        damping: 15
      }
    }
  };

  return (
    <section className="py-24 px-6 bg-background relative overflow-hidden">
      <div className="max-w-6xl mx-auto">
        <motion.div
          initial={{ y: 30, opacity: 0 }}
          whileInView={{ y: 0, opacity: 1 }}
          viewport={{ once: true }}
          transition={{ duration: 0.5 }}
        >
          <p className="text-secondary font-bold tracking-widest uppercase text-sm mb-3">Supported Exercises</p>
          <h2 className="font-heading text-5xl md:text-6xl mb-16 comic-shadow">
            Master These<br />HERO MOVES!
          </h2>
        </motion.div>
        
        <motion.div 
          variants={containerVariants}
          initial="hidden"
          whileInView="visible"
          viewport={{ once: true }}
          className="space-y-6"
        >
          {exercises.map((ex) => (
            <motion.div
              key={ex.name}
              variants={itemVariants}
              whileHover={{ scale: 1.02, x: 10 }}
              className="flex flex-col md:flex-row md:items-center justify-between bg-card rounded-2xl p-6 md:p-8 border border-primary/30 cursor-pointer group"
              style={{
                boxShadow: "0 0 15px hsl(50 35% 45% / 0.1), 4px 4px 0px hsl(345 50% 35% / 0.5)"
              }}
            >
              <div className="flex items-center gap-6 mb-4 md:mb-0">
                <motion.div
                  whileHover={{ rotate: 360 }}
                  transition={{ duration: 0.5 }}
                  className="w-12 h-12 bg-secondary/80 rounded-xl flex items-center justify-center border border-secondary-foreground/20"
                >
                  <ex.icon className="w-6 h-6 text-secondary-foreground" />
                </motion.div>
                <h3 className="font-heading text-4xl md:text-5xl tracking-wide">{ex.name}</h3>
                <span className="text-xs tracking-widest uppercase text-secondary-foreground bg-secondary/80 px-4 py-2 rounded-full font-bold border border-secondary-foreground/20">
                  {ex.tag}
                </span>
              </div>
              <div className="flex items-center gap-6">
                <p className="text-muted-foreground font-semibold max-w-xs">{ex.description}</p>
                <motion.div
                  whileHover={{ x: 10 }}
                  className="shrink-0"
                >
                  <ArrowRight className="w-6 h-6 text-foreground group-hover:text-secondary transition-colors" />
                </motion.div>
              </div>
            </motion.div>
          ))}
        </motion.div>
      </div>
    </section>
  );
};

export default Exercises;
