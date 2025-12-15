#!/bin/bash
# =============================================================================
# Security Scanning Script
# Runs Snyk scans for SCA, SAST, IaC, Container, and AIBOM
# =============================================================================
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
OUTPUT_DIR="$PROJECT_ROOT/security-reports"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

log_info() { echo -e "${BLUE}[INFO]${NC} $1"; }
log_success() { echo -e "${GREEN}[SUCCESS]${NC} $1"; }
log_warning() { echo -e "${YELLOW}[WARNING]${NC} $1"; }
log_error() { echo -e "${RED}[ERROR]${NC} $1"; }

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Check Snyk CLI
check_snyk() {
    if ! command -v snyk &>/dev/null; then
        log_error "Snyk CLI not found. Please install it first."
        echo "  npm install -g snyk"
        echo "  OR"
        echo "  brew install snyk"
        exit 1
    fi
    
    # Check authentication
    if ! snyk auth check &>/dev/null; then
        log_warning "Snyk not authenticated. Running 'snyk auth'..."
        snyk auth
    fi
}

run_sca_scan() {
    log_info "Running SCA (Software Composition Analysis) scan..."
    
    cd "$PROJECT_ROOT"
    snyk test \
        --all-projects \
        --json-file-output="$OUTPUT_DIR/sca-report.json" \
        --severity-threshold=medium || true
    
    log_success "SCA scan complete: $OUTPUT_DIR/sca-report.json"
}

run_code_scan() {
    log_info "Running SAST (Static Application Security Testing) scan..."
    
    cd "$PROJECT_ROOT"
    snyk code test \
        --json-file-output="$OUTPUT_DIR/sast-report.json" \
        --severity-threshold=medium || true
    
    log_success "SAST scan complete: $OUTPUT_DIR/sast-report.json"
}

run_iac_scan() {
    log_info "Running IaC (Infrastructure as Code) scan..."
    
    cd "$PROJECT_ROOT"
    snyk iac test \
        --json-file-output="$OUTPUT_DIR/iac-report.json" \
        --severity-threshold=medium || true
    
    log_success "IaC scan complete: $OUTPUT_DIR/iac-report.json"
}

run_container_scan() {
    log_info "Running Container security scan..."
    
    # Build the image first
    cd "$PROJECT_ROOT"
    docker build -t aibom-app:scan . 2>/dev/null || {
        log_warning "Docker build failed, skipping container scan"
        return
    }
    
    snyk container test aibom-app:scan \
        --json-file-output="$OUTPUT_DIR/container-report.json" \
        --severity-threshold=medium || true
    
    log_success "Container scan complete: $OUTPUT_DIR/container-report.json"
}

run_aibom_scan() {
    log_info "Running AIBOM (AI Bill of Materials) scan..."
    
    cd "$PROJECT_ROOT"
    snyk aibom \
        --json > "$OUTPUT_DIR/aibom-report.json" 2>/dev/null || true
    
    if [ -s "$OUTPUT_DIR/aibom-report.json" ]; then
        log_success "AIBOM scan complete: $OUTPUT_DIR/aibom-report.json"
    else
        log_warning "AIBOM scan produced no output (feature may be experimental)"
    fi
}

generate_summary() {
    log_info "Generating summary report..."
    
    cat > "$OUTPUT_DIR/summary.md" << EOF
# Security Scan Summary

**Generated:** $(date -u +"%Y-%m-%d %H:%M:%S UTC")
**Project:** aibom-test

## Scan Results

| Scan Type | Status | Report |
|-----------|--------|--------|
| SCA (Open Source) | $([ -f "$OUTPUT_DIR/sca-report.json" ] && echo "✅ Complete" || echo "❌ Failed") | [View](./sca-report.json) |
| SAST (Code) | $([ -f "$OUTPUT_DIR/sast-report.json" ] && echo "✅ Complete" || echo "❌ Failed") | [View](./sast-report.json) |
| IaC | $([ -f "$OUTPUT_DIR/iac-report.json" ] && echo "✅ Complete" || echo "❌ Failed") | [View](./iac-report.json) |
| Container | $([ -f "$OUTPUT_DIR/container-report.json" ] && echo "✅ Complete" || echo "❌ Failed/Skipped") | [View](./container-report.json) |
| AIBOM | $([ -f "$OUTPUT_DIR/aibom-report.json" ] && echo "✅ Complete" || echo "❌ Failed") | [View](./aibom-report.json) |

## Quick Stats

EOF

    # Add vulnerability counts if reports exist
    if [ -f "$OUTPUT_DIR/sca-report.json" ]; then
        echo "### SCA Vulnerabilities" >> "$OUTPUT_DIR/summary.md"
        python3 -c "
import json
try:
    with open('$OUTPUT_DIR/sca-report.json') as f:
        data = json.load(f)
        if isinstance(data, list):
            total = sum(len(p.get('vulnerabilities', [])) for p in data)
        else:
            total = len(data.get('vulnerabilities', []))
        print(f'- Total vulnerabilities: {total}')
except:
    print('- Unable to parse report')
" >> "$OUTPUT_DIR/summary.md"
    fi

    log_success "Summary generated: $OUTPUT_DIR/summary.md"
}

# Main
main() {
    echo "=========================================="
    echo "Security Scanning Suite"
    echo "=========================================="
    
    check_snyk
    
    run_sca_scan
    run_code_scan
    run_iac_scan
    run_container_scan
    run_aibom_scan
    
    generate_summary
    
    echo ""
    log_success "All scans complete!"
    echo "Reports saved to: $OUTPUT_DIR"
    echo ""
    echo "View summary: cat $OUTPUT_DIR/summary.md"
}

main "$@"

