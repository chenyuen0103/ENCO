mkdir -p experiments/responses_old/sachs
rm experiments/responses/sachs/*.json
rm experiments/responses/sachs/*.pdf
mv experiments/responses/sachs/*grpo* experiments/responses_old/sachs/
mv experiments/responses/sachs/*stage*  experiments/responses_old/sachs/
mv experiments/responses/sachs/*checkpoint-4*  experiments/responses_old/sachs/
mv experiments/responses/sachs/*sft*  experiments/responses_old/sachs/