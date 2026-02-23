#!/bin/bash

# Comprehensive Analysis Runner for Speaker Verification
# Generates all plots and metrics in one command

# Default paths
EPOCH=${EPOCH:-1}
EXP_DIR=exp/fusion_wav2vec2_samresnet34_voxblink_ft_tidy1
SCORE_FILE=${EXP_DIR}/scores/trials.kaldi.score
TRIALS_FILE=data/tidyvoice_dev/trials/trials.kaldi

# Parse command line arguments
HELP_MSG="Usage: $0 [OPTIONS]

Generate comprehensive speaker verification analysis plots and metrics

OPTIONS:
  --epoch N             Specify epoch number (default: 1)
  --score_file FILE     Path to score file (default: exp/.../scores/trials.kaldi.score)
  --trials_file FILE    Path to trials file (default: data/.../trials.kaldi)
  --exp_dir DIR         Experiment directory (default: exp/samresnet34_voxblink_ft_tidy)
  --skip-category       Skip category analysis (faster, only DET/ROC curves)
  -h, --help           Show this help message

EXAMPLES:
  # Analyze epoch 1 with full category analysis
  $0 --epoch 1

  # Quick analysis without categories
  $0 --epoch 2 --skip-category

  # Analyze custom score file
  $0 --score_file path/to/custom.score --trials_file path/to/trials

OUTPUT:
  All plots and metrics will be saved in the same directory as the score file:
  - evaluation_curves.png: DET and ROC curves
  - score_distributions_by_category.png: 4 category plots
  - score_distributions_combined.png: Combined smooth curves
  - category_metrics.txt: Detailed metrics file
"

SKIP_CATEGORY=false

while [[ $# -gt 0 ]]; do
    case $1 in
        --epoch)
            EPOCH="$2"
            shift 2
            ;;
        --score_file)
            SCORE_FILE="$2"
            shift 2
            ;;
        --trials_file)
            TRIALS_FILE="$2"
            shift 2
            ;;
        --exp_dir)
            EXP_DIR="$2"
            SCORE_FILE="${EXP_DIR}/scores/trials.kaldi.score"
            shift 2
            ;;
        --skip-category)
            SKIP_CATEGORY=true
            shift
            ;;
        -h|--help)
            echo "$HELP_MSG"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Use -h or --help for usage information"
            exit 1
            ;;
    esac
done

# Update score file path if epoch-specific
if [ ! -f "${SCORE_FILE}" ]; then
    SCORE_FILE="${EXP_DIR}/scores/epoch_${EPOCH}_trials.kaldi.score"
fi

echo "=============================================================================="
echo "COMPREHENSIVE SPEAKER VERIFICATION ANALYSIS"
echo "=============================================================================="
echo "Configuration:"
echo "  Epoch:        ${EPOCH}"
echo "  Score file:   ${SCORE_FILE}"
echo "  Trials file:  ${TRIALS_FILE}"
echo "  Exp dir:      ${EXP_DIR}"
echo "  Skip category: ${SKIP_CATEGORY}"
echo "=============================================================================="
echo ""

# Check if score file exists
if [ ! -f "${SCORE_FILE}" ]; then
    echo "Error: Score file not found: ${SCORE_FILE}"
    echo ""
    echo "Available score files:"
    find ${EXP_DIR}/scores -name "*.score" 2>/dev/null || echo "  None found"
    exit 1
fi

# Check if trials file exists (needed for category analysis)
if [ "${SKIP_CATEGORY}" = false ] && [ ! -f "${TRIALS_FILE}" ]; then
    echo "Warning: Trials file not found: ${TRIALS_FILE}"
    echo "Proceeding with basic analysis only (no category breakdown)"
    SKIP_CATEGORY=true
fi

# Run the comprehensive analysis
echo "Running analysis..."
echo ""

if [ "${SKIP_CATEGORY}" = true ]; then
    python comprehensive_analysis.py "${SCORE_FILE}"
else
    python comprehensive_analysis.py "${SCORE_FILE}" "${TRIALS_FILE}"
fi

EXIT_CODE=$?

if [ $EXIT_CODE -eq 0 ]; then
    echo ""
    echo "=============================================================================="
    echo "✓ Analysis completed successfully!"
    echo "=============================================================================="
    echo "Output directory: $(dirname ${SCORE_FILE})/"
    echo ""
    echo "Generated files:"
    ls -lh $(dirname ${SCORE_FILE})/*.png 2>/dev/null | awk '{print "  " $9 " (" $5 ")"}'
    if [ -f "$(dirname ${SCORE_FILE})/category_metrics.txt" ]; then
        echo "  $(dirname ${SCORE_FILE})/category_metrics.txt"
    fi
    echo ""
else
    echo ""
    echo "=============================================================================="
    echo "✗ Analysis failed with exit code: $EXIT_CODE"
    echo "=============================================================================="
    exit $EXIT_CODE
fi
