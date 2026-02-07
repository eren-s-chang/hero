import Hero from "@/components/Hero";
import HowItWorks from "@/components/HowItWorks";
import Exercises from "@/components/Exercises";
import Footer from "@/components/Footer";

const Index = () => {
  return (
    <main className="bg-background text-foreground min-h-screen">
      <Hero />
      <HowItWorks />
      <Exercises />
      <Footer />
    </main>
  );
};

export default Index;
