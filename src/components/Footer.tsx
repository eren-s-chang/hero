import { motion } from "framer-motion";

const Footer = () => (
  <motion.footer 
    initial={{ opacity: 0 }}
    whileInView={{ opacity: 1 }}
    viewport={{ once: true }}
    className="border-t border-border py-12 px-6 md:px-12 bg-background"
  >
    <div className="max-w-6xl mx-auto flex flex-col md:flex-row items-center justify-between gap-4">
      <span className="font-heading text-2xl tracking-widest text-foreground">
        FORMGUARD
      </span>
      <p className="text-muted-foreground text-sm">
        © 2026 FormGuard. Precision movement analysis.
      </p>
    </div>
  </motion.footer>
);

export default Footer;
