from yeardistribution import YearDistribution
import os

KEY = os.environ.get("OPENAI_API_KEY")

answer = YearDistribution("gpt-4o-mini", KEY)

print(answer.Year("제작년 행사는 어땠어?"))


