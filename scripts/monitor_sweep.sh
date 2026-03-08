#!/bin/bash
# Monitor the main sensitivity sweep (PID 98797) and launch extended sweep when done

OUTPUT_FILE="/private/tmp/claude-501/-Users-jasonjiao-Downloads-SleepResearch/tasks/b1ec6j8tf.output"
EXTENDED_SCRIPT="/Users/jasonjiao/Downloads/SleepResearch/scripts/run_so_gating_sensitivity_extended.py"
LOG_FILE="/Users/jasonjiao/Downloads/SleepResearch/results/so_gating_study/monitor.log"

echo "$(date): Monitor started. Watching PID 98797..." >> "$LOG_FILE"

while kill -0 98797 2>/dev/null; do
    sleep 300  # check every 5 min
done

echo "$(date): Main sweep (PID 98797) COMPLETED." >> "$LOG_FILE"

# Grab final results
echo "$(date): Final output tail:" >> "$LOG_FILE"
tail -50 "$OUTPUT_FILE" >> "$LOG_FILE" 2>/dev/null

# Check if crossover happened at 1.0
LAST_GAP=$(grep "continuous_pulsed_gap" /Users/jasonjiao/Downloads/SleepResearch/results/so_gating_study/sensitivity_summary.csv 2>/dev/null | tail -1)
echo "$(date): Last gap line: $LAST_GAP" >> "$LOG_FILE"

# Launch extended sweep (1.5, 2.0) automatically
echo "$(date): Launching extended sweep (so_modulation=1.5, 2.0)..." >> "$LOG_FILE"
cd /Users/jasonjiao/Downloads/SleepResearch
python "$EXTENDED_SCRIPT" --workers 6 --n-subjects 208 >> "$LOG_FILE" 2>&1

echo "$(date): Extended sweep COMPLETED." >> "$LOG_FILE"
echo "DONE" >> "$LOG_FILE"
