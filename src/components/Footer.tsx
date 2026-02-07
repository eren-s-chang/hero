import { motion } from "framer-motion";
import { Zap } from "lucide-react";

const Footer = () => (
  <motion.footer 
    initial={{ opacity: 0 }}
    whileInView={{ opacity: 1 }}
    viewport={{ once: true }}
    className="border-t-4 border-foreground py-12 px-6 bg-primary"
  >
    <div className="max-w-6xl mx-auto flex flex-col md:flex-row items-center justify-between gap-4">
      <motion.span 
        whileHover={{ scale: 1.1, rotate: -3 }}
        className="font-heading text-3xl flex items-center gap-2 text-foreground"
      >
        <Zap className="w-6 h-6 fill-foreground" />
        FormGuard
      </motion.span>
      <p className="text-foreground font-bold">
        © 2026 FormGuard. Train like a hero!
      </p>
    </div>
  </motion.footer>
);

export default Footer;
