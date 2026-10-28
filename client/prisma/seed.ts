import { PrismaClient } from "@prisma/client";

const prisma = new PrismaClient();

const rand = (min: number, max: number): number => min + Math.random() * (max - min);
const round1 = (n: number): number => Math.round(n * 10) / 10;

async function main(): Promise<void> {
  await prisma.workoutLog.deleteMany();
  await prisma.user.deleteMany();

  const user = await prisma.user.create({
    data: { email: "demo@example.com", sexBinary: 1, age: 27, bodyweightKg: 85, benchBaseline1rmKg: 120 },
  });

  const logs = [];
  for (let day = 29; day >= 0; day -= 2) {
    const date = new Date();
    date.setDate(date.getDate() - day);
    const incline = day % 4 === 1;
    const meanRpe = rand(6.5, 9.5);
    logs.push({
      userId: user.id,
      date,
      exercise: incline ? "Incline Bench Press" : "Bench Press",
      sessionVolumeKg: Math.round(user.benchBaseline1rmKg * (incline ? 0.8 : 1) * rand(8, 18)),
      meanRpe: round1(meanRpe),
      fatigueDropOff: round1(Math.max(0, 6 * (meanRpe - 6) + rand(-4, 8))),
    });
  }
  await prisma.workoutLog.createMany({ data: logs });
  console.log(`Seeded ${logs.length} sessions over 30 days for ${user.email}`);
}

main().finally(() => prisma.$disconnect());
