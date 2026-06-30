import { MMOULauncher } from "@/components/benchmark/mmou-launcher";

export default function BenchmarkDashboard() {
  return (
    <div className="flex min-h-screen min-w-0 flex-col">
      <div className="flex-1 min-w-0 space-y-4 p-4">
        <MMOULauncher />
      </div>
    </div>
  );
}
