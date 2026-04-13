"""
Report generator for creating comprehensive exam violation reports.
Generates PDF and HTML reports with visualizations.
"""

import os
import matplotlib
matplotlib.use('Agg')  # Set non-interactive backend
import matplotlib.pyplot as plt
from jinja2 import Environment, FileSystemLoader
from datetime import datetime
from typing import Dict, Any, List, Optional
import logging

try:
    import pdfkit
    PDFKIT_AVAILABLE = True
except ImportError:
    PDFKIT_AVAILABLE = False
    pdfkit = None
    print("⚠️ pdfkit not available - PDF reports disabled. Install with: pip install pdfkit")


class ReportGenerator:
    """Generates comprehensive violation reports."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize report generator.
        
        Args:
            config: Configuration dictionary (optional)
        """
        if config:
            self.config = config.get('reporting', {})
        else:
            self.config = {}
        
        self.output_dir = self.config.get('output_dir', './reports/generated')
        self.image_dir = os.path.join(self.output_dir, 'images')
        
        # Set up directories
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(self.image_dir, exist_ok=True)
        
        # Configure templates
        template_path = os.path.join(os.path.dirname(__file__), '..', 'templates')
        if not os.path.exists(template_path):
            template_path = os.path.join(os.path.dirname(__file__), '..', '..', 'templates')
        self.template_env = Environment(loader=FileSystemLoader(template_path))
        
        # Configure logging
        self.logger = logging.getLogger('ReportGenerator')
        self.logger.setLevel(logging.INFO)
        
        # Severity mapping (from config, with defaults)
        default_severity = {
            'FACE_DISAPPEARED': 1,
            'FACE_MISSING': 1,
            'GAZE_AWAY': 2,
            'GAZE_OFFSCREEN': 2,
            'EYES_CLOSED': 2,
            'FACE_TURNED': 2,
            'FACE_BLURRY': 2,
            'MOUTH_MOVING': 3,
            'TALKING_DETECTED': 3,
            'AUDIO_DETECTED': 3,
            'VOICE_DETECTED': 3,
            'MULTIPLE_FACES': 4,
            'OBJECT_DETECTED': 5,
            'IDENTITY_MISMATCH': 1,
            'TAB_SWITCH': 4,
            'WINDOW_BLUR': 4,
        }
        config_severity = self.config.get('severity_levels', {})
        # Convert config keys to uppercase and merge with defaults
        self.severity_map = {}
        for key, value in default_severity.items():
            self.severity_map[key] = config_severity.get(key, value)
        # Also add any additional keys from config
        for key, value in config_severity.items():
            if key not in self.severity_map:
                self.severity_map[key] = value
    
    def generate_report(self, student_info: Dict[str, Any], violations: List[Dict[str, Any]], 
                       output_format: str = 'html') -> Optional[str]:
        """
        Generate a comprehensive exam violation report.
        
        Args:
            student_info: Student identification data
            violations: List of violation dictionaries
            output_format: 'pdf' or 'html'
            
        Returns:
            Path to generated report file or None if failed
        """
        try:
            # Prepare report data
            report_data = {
                'student': student_info,
                'violations': violations,
                'generated_at': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                'stats': self._calculate_stats(violations),
                'timeline_image': self._generate_timeline(violations, student_info.get('id', 'unknown')),
                'heatmap_image': self._generate_heatmap(violations, student_info.get('id', 'unknown')),
                'has_images': False
            }
            
            # Check if we have images to include
            if report_data['timeline_image'] or report_data['heatmap_image']:
                report_data['has_images'] = True
            
            # Render HTML template
            template = self.template_env.get_template('base_report.html')
            html_content = template.render(report_data)
            
            # Generate output filename
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            student_id = student_info.get('id', 'unknown')
            filename = f"report_{student_id}_{timestamp}"
            output_path = os.path.join(self.output_dir, f"{filename}.{output_format.lower()}")
            
            # Generate the chosen output format
            if output_format.lower() == 'pdf' and PDFKIT_AVAILABLE:
                options = {
                    'enable-local-file-access': None,
                    'quiet': '',
                    'margin-top': '10mm',
                    'margin-right': '10mm',
                    'margin-bottom': '10mm',
                    'margin-left': '10mm'
                }
                pdfkit_config = self.config.get('wkhtmltopdf_path')
                if pdfkit_config:
                    config = pdfkit.configuration(wkhtmltopdf=pdfkit_config)
                else:
                    config = None
                pdfkit.from_string(html_content, output_path, options=options, configuration=config)
            else:
                # Save as HTML
                with open(output_path, 'w', encoding='utf-8') as f:
                    f.write(html_content)
            
            self.logger.info(f"Report generated at: {output_path}")
            return output_path
        
        except Exception as e:
            self.logger.error(f"Failed to generate report: {str(e)}")
            return None
    
    def _calculate_stats(self, violations: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Calculate summary statistics from violations."""
        stats = {
            'total': len(violations),
            'by_type': {},
            'timeline': [],
            'severity_score': 0
        }
        
        for violation in violations:
            violation_type = violation.get('type', 'UNKNOWN')
            # Count by type
            stats['by_type'][violation_type] = stats['by_type'].get(violation_type, 0) + 1
            
            # Add to timeline
            timestamp_str = violation.get('timestamp', datetime.now().strftime("%Y%m%d_%H%M%S_%f"))
            try:
                timestamp = datetime.strptime(timestamp_str, "%Y%m%d_%H%M%S_%f")
            except ValueError:
                try:
                    timestamp = datetime.strptime(timestamp_str, "%Y-%m-%d %H:%M:%S")
                except ValueError:
                    timestamp = datetime.now()
            
            stats['timeline'].append({
                'time': timestamp,
                'type': violation_type,
                'severity': self.severity_map.get(violation_type, 1)
            })
            
            # Calculate total severity score
            stats['severity_score'] += self.severity_map.get(violation_type, 1)
        
        # Calculate average severity
        if stats['total'] > 0:
            stats['average_severity'] = stats['severity_score'] / stats['total']
        else:
            stats['average_severity'] = 0
        
        return stats
    
    def _generate_timeline(self, violations: List[Dict[str, Any]], student_id: str) -> Optional[str]:
        """Generate violation timeline visualization."""
        if not violations:
            return None
        
        try:
            # Prepare data
            times = []
            severities = []
            labels = []
            
            for violation in violations:
                timestamp_str = violation.get('timestamp', datetime.now().strftime("%Y%m%d_%H%M%S_%f"))
                try:
                    timestamp = datetime.strptime(timestamp_str, "%Y%m%d_%H%M%S_%f")
                except ValueError:
                    try:
                        timestamp = datetime.strptime(timestamp_str, "%Y-%m-%d %H:%M:%S")
                    except ValueError:
                        timestamp = datetime.now()
                
                times.append(timestamp)
                violation_type = violation.get('type', 'UNKNOWN')
                severities.append(self.severity_map.get(violation_type, 1))
                labels.append(violation_type)
            
            # Create figure
            plt.figure(figsize=(12, 5))
            ax = plt.gca()
            
            # Plot timeline
            plt.plot(times, severities, 'o-', markersize=8)
            
            # Add labels
            for i, (time, severity, label) in enumerate(zip(times, severities, labels)):
                plt.annotate(
                    label,
                    (time, severity),
                    textcoords="offset points",
                    xytext=(0, 10),
                    ha='center',
                    fontsize=8
                )
            
            # Format plot
            plt.title(f"Violation Timeline - {student_id}")
            plt.xlabel("Time")
            plt.ylabel("Severity Level")
            plt.grid(True, linestyle='--', alpha=0.7)
            plt.xticks(rotation=45)
            plt.tight_layout()
            
            # Save image
            timeline_path = os.path.join(self.image_dir, f'timeline_{student_id}.png')
            plt.savefig(timeline_path, dpi=150, bbox_inches='tight')
            plt.close()
            
            return timeline_path
        
        except Exception as e:
            self.logger.error(f"Failed to generate timeline: {str(e)}")
            return None
    
    def _generate_heatmap(self, violations: List[Dict[str, Any]], student_id: str) -> Optional[str]:
        """Generate violation frequency heatmap."""
        if not violations:
            return None
        
        try:
            # Count violations by type
            violation_counts = {}
            for violation in violations:
                violation_type = violation.get('type', 'UNKNOWN')
                violation_counts[violation_type] = violation_counts.get(violation_type, 0) + 1
            
            # Sort by count
            sorted_types = sorted(violation_counts.items(), key=lambda x: x[1], reverse=True)
            types, counts = zip(*sorted_types) if sorted_types else ([], [])
            
            # Create figure
            plt.figure(figsize=(10, 5))
            
            # Create colormap based on severity
            colors = [plt.cm.Reds(self.severity_map.get(t, 1)/5) for t in types]
            
            # Plot horizontal bars
            bars = plt.barh(
                types,
                counts,
                color=colors,
                edgecolor='black',
                linewidth=0.7
            )
            
            # Add count labels
            for bar in bars:
                width = bar.get_width()
                plt.text(
                    width + 0.3,
                    bar.get_y() + bar.get_height()/2,
                    f"{int(width)}",
                    va='center',
                    ha='left',
                    fontsize=10
                )
            
            # Format plot
            plt.title(f"Violation Frequency - {student_id}")
            plt.xlabel("Count")
            plt.ylabel("Violation Type")
            plt.grid(True, linestyle='--', alpha=0.3, axis='x')
            plt.tight_layout()
            
            # Save image
            heatmap_path = os.path.join(self.image_dir, f'heatmap_{student_id}.png')
            plt.savefig(heatmap_path, dpi=150, bbox_inches='tight')
            plt.close()
            
            return heatmap_path
        
        except Exception as e:
            self.logger.error(f"Failed to generate heatmap: {str(e)}")
            return None

