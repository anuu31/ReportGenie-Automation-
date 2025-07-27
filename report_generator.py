import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from reportlab.lib.pagesizes import A4
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle, PageBreak
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY
from datetime import datetime
import os
import json
import warnings
import logging

# Configure settings
warnings.filterwarnings('ignore')
plt.style.use('default')
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class DataAnalyzer:
    """Handles all data analysis operations"""
    
    def __init__(self, data):
        self.data = data
        self.analysis_results = {}
        self.numeric_columns = data.select_dtypes(include=[np.number]).columns.tolist()
        self.categorical_columns = data.select_dtypes(include=['object']).columns.tolist()
    
    def basic_statistics(self):
        """Calculate comprehensive statistical measures"""
        stats = {}
        
        for col in self.numeric_columns:
            try:
                col_data = self.data[col].dropna()
                stats[col] = {
                    'count': len(col_data),
                    'mean': round(col_data.mean(), 3),
                    'median': round(col_data.median(), 3),
                    'std': round(col_data.std(), 3),
                    'min': round(col_data.min(), 3),
                    'max': round(col_data.max(), 3),
                    'q25': round(col_data.quantile(0.25), 3),
                    'q75': round(col_data.quantile(0.75), 3),
                    'null_count': self.data[col].isnull().sum(),
                    'null_percentage': round((self.data[col].isnull().sum() / len(self.data)) * 100, 2)
                }
            except Exception as e:
                logger.warning(f"Error calculating stats for {col}: {e}")
                continue
        
        self.analysis_results['basic_stats'] = stats
        return stats
    
    def correlation_analysis(self):
        """Perform correlation analysis"""
        if len(self.numeric_columns) > 1:
            try:
                correlation_matrix = self.data[self.numeric_columns].corr()
                self.analysis_results['correlation'] = correlation_matrix
                return correlation_matrix
            except Exception as e:
                logger.warning(f"Error in correlation analysis: {e}")
        return None
    
    def categorical_analysis(self):
        """Analyze categorical variables"""
        cat_stats = {}
        
        for col in self.categorical_columns:
            try:
                value_counts = self.data[col].value_counts()
                cat_stats[col] = {
                    'unique_count': self.data[col].nunique(),
                    'top_values': dict(value_counts.head(5)),
                    'null_count': self.data[col].isnull().sum(),
                    'null_percentage': round((self.data[col].isnull().sum() / len(self.data)) * 100, 2)
                }
            except Exception as e:
                logger.warning(f"Error analyzing categorical column {col}: {e}")
                continue
        
        self.analysis_results['categorical'] = cat_stats
        return cat_stats
    
    def data_quality_assessment(self):
        """Assess overall data quality"""
        total_cells = len(self.data) * len(self.data.columns)
        missing_cells = self.data.isnull().sum().sum()
        
        quality_metrics = {
            'total_records': len(self.data),
            'total_columns': len(self.data.columns),
            'total_cells': total_cells,
            'missing_cells': missing_cells,
            'missing_percentage': round((missing_cells / total_cells) * 100, 2),
            'duplicate_rows': self.data.duplicated().sum(),
            'numeric_columns': len(self.numeric_columns),
            'categorical_columns': len(self.categorical_columns),
            'memory_usage_mb': round(self.data.memory_usage(deep=True).sum() / 1024**2, 2)
        }
        
        self.analysis_results['data_quality'] = quality_metrics
        return quality_metrics

class ChartGenerator:
    """Generates visualizations for the report"""
    
    def __init__(self, data, output_dir="temp_charts"):
        self.data = data
        self.output_dir = output_dir
        self.chart_files = []
        
        # Create output directory
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
    
    def create_distribution_plots(self):
        """Create distribution plots for numeric columns"""
        numeric_cols = self.data.select_dtypes(include=[np.number]).columns[:4]  # Limit to 4
        
        if len(numeric_cols) == 0:
            return None
        
        try:
            fig, axes = plt.subplots(2, 2, figsize=(12, 10))
            fig.suptitle('Distribution Analysis', fontsize=16, fontweight='bold')
            
            for i, col in enumerate(numeric_cols):
                if i >= 4:  # Safety check
                    break
                    
                row, col_idx = i // 2, i % 2
                
                # Handle potential issues with data
                data_to_plot = self.data[col].dropna()
                if len(data_to_plot) == 0:
                    continue
                
                axes[row, col_idx].hist(data_to_plot, bins=30, alpha=0.7, 
                                      color='skyblue', edgecolor='black')
                axes[row, col_idx].set_title(f'Distribution of {col}')
                axes[row, col_idx].set_xlabel(col)
                axes[row, col_idx].set_ylabel('Frequency')
                axes[row, col_idx].grid(True, alpha=0.3)
            
            # Hide empty subplots
            for i in range(len(numeric_cols), 4):
                row, col_idx = i // 2, i % 2
                axes[row, col_idx].set_visible(False)
            
            plt.tight_layout()
            filename = f"{self.output_dir}/distribution_plots.png"
            plt.savefig(filename, dpi=300, bbox_inches='tight')
            plt.close()
            
            self.chart_files.append(filename)
            return filename
            
        except Exception as e:
            logger.error(f"Error creating distribution plots: {e}")
            plt.close()
            return None
    
    def create_correlation_heatmap(self, correlation_matrix):
        """Create correlation heatmap"""
        if correlation_matrix is None or correlation_matrix.empty:
            return None
        
        try:
            plt.figure(figsize=(10, 8))
            mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))
            
            sns.heatmap(correlation_matrix, mask=mask, annot=True, cmap='coolwarm', 
                       center=0, square=True, linewidths=0.5, cbar_kws={"shrink": .8})
            
            plt.title('Correlation Matrix', fontsize=16, fontweight='bold')
            plt.xticks(rotation=45, ha='right')
            plt.yticks(rotation=0)
            plt.tight_layout()
            
            filename = f"{self.output_dir}/correlation_heatmap.png"
            plt.savefig(filename, dpi=300, bbox_inches='tight')
            plt.close()
            
            self.chart_files.append(filename)
            return filename
            
        except Exception as e:
            logger.error(f"Error creating correlation heatmap: {e}")
            plt.close()
            return None
    
    def create_categorical_charts(self):
        """Create charts for categorical data"""
        categorical_cols = self.data.select_dtypes(include=['object']).columns[:2]  # Limit to 2
        
        if len(categorical_cols) == 0:
            return None
        
        try:
            fig, axes = plt.subplots(1, len(categorical_cols), figsize=(6*len(categorical_cols), 6))
            if len(categorical_cols) == 1:
                axes = [axes]
            
            fig.suptitle('Categorical Data Analysis', fontsize=16, fontweight='bold')
            
            for i, col in enumerate(categorical_cols):
                value_counts = self.data[col].value_counts().head(8)  # Top 8 values
                
                bars = axes[i].bar(range(len(value_counts)), value_counts.values, 
                                 color='lightcoral', alpha=0.8)
                axes[i].set_title(f'Distribution of {col}')
                axes[i].set_xlabel(col)
                axes[i].set_ylabel('Count')
                axes[i].set_xticks(range(len(value_counts)))
                axes[i].set_xticklabels(value_counts.index, rotation=45, ha='right')
                axes[i].grid(True, alpha=0.3)
                
                # Add value labels on bars
                for bar in bars:
                    height = bar.get_height()
                    axes[i].text(bar.get_x() + bar.get_width()/2., height,
                               f'{int(height)}', ha='center', va='bottom')
            
            plt.tight_layout()
            filename = f"{self.output_dir}/categorical_charts.png"
            plt.savefig(filename, dpi=300, bbox_inches='tight')
            plt.close()
            
            self.chart_files.append(filename)
            return filename
            
        except Exception as e:
            logger.error(f"Error creating categorical charts: {e}")
            plt.close()
            return None
    
    def cleanup_charts(self):
        """Clean up temporary chart files"""
        for chart_file in self.chart_files:
            try:
                if os.path.exists(chart_file):
                    os.remove(chart_file)
            except Exception as e:
                logger.warning(f"Could not remove {chart_file}: {e}")
        
        # Remove directory if empty
        try:
            if os.path.exists(self.output_dir) and not os.listdir(self.output_dir):
                os.rmdir(self.output_dir)
        except Exception as e:
            logger.warning(f"Could not remove directory {self.output_dir}: {e}")

class PDFReportGenerator:
    """Generates professional PDF reports"""
    
    def __init__(self, filename="Automated_Analysis_Report.pdf"):
        self.filename = filename
        self.story = []
        self.styles = getSampleStyleSheet()
        self._setup_custom_styles()
    
    def _setup_custom_styles(self):
        """Setup custom PDF styles"""
        # Title style
        self.title_style = ParagraphStyle(
            'CustomTitle',
            parent=self.styles['Title'],
            fontSize=24,
            textColor=colors.darkblue,
            spaceAfter=30,
            alignment=TA_CENTER,
            fontName='Helvetica-Bold'
        )
        
        # Heading styles
        self.heading_style = ParagraphStyle(
            'CustomHeading',
            parent=self.styles['Heading1'],
            fontSize=16,
            textColor=colors.darkblue,
            spaceBefore=20,
            spaceAfter=12,
            fontName='Helvetica-Bold'
        )
        
        self.subheading_style = ParagraphStyle(
            'CustomSubHeading',
            parent=self.styles['Heading2'],
            fontSize=14,
            textColor=colors.darkgreen,
            spaceBefore=15,
            spaceAfter=8,
            fontName='Helvetica-Bold'
        )
        
        # Body style
        self.body_style = ParagraphStyle(
            'CustomBody',
            parent=self.styles['Normal'],
            fontSize=11,
            spaceAfter=8,
            alignment=TA_JUSTIFY,
            fontName='Helvetica'
        )
    
    def add_title_page(self):
        """Add professional title page"""
        # Title
        self.story.append(Spacer(1, 2*inch))
        self.story.append(Paragraph("AUTOMATED DATA ANALYSIS REPORT", self.title_style))
        self.story.append(Spacer(1, 0.5*inch))
        
        # Subtitle
        subtitle_style = ParagraphStyle('Subtitle', parent=self.styles['Normal'],
                                      fontSize=16, alignment=TA_CENTER, 
                                      textColor=colors.darkred)
        self.story.append(Paragraph("Comprehensive Statistical Analysis & Insights", subtitle_style))
        self.story.append(Spacer(1, 1*inch))
        
        # Metadata
        current_date = datetime.now().strftime("%B %d, %Y at %I:%M %p")
        self.story.append(Paragraph(f"<b>Generated:</b> {current_date}", self.body_style))
        self.story.append(Paragraph("<b>System:</b> CODTECH Automated Report Generator", self.body_style))
        self.story.append(Paragraph("<b>Version:</b> 2.0", self.body_style))
        
        self.story.append(PageBreak())
    
    def add_executive_summary(self, data_quality):
        """Add executive summary"""
        self.story.append(Paragraph("EXECUTIVE SUMMARY", self.heading_style))
        
        summary_text = f"""
        This automated report presents a comprehensive analysis of the provided dataset containing 
        {data_quality['total_records']:,} records across {data_quality['total_columns']} variables. 
        The dataset includes {data_quality['numeric_columns']} numeric and {data_quality['categorical_columns']} 
        categorical variables, with {data_quality['missing_percentage']}% missing data overall.
        
        Key findings include statistical summaries, correlation patterns, data quality metrics, 
        and actionable insights derived through automated analysis. This report serves as a 
        foundation for data-driven decision making and identifies areas requiring further investigation.
        """
        
        self.story.append(Paragraph(summary_text, self.body_style))
        self.story.append(Spacer(1, 20))
    
    def add_data_overview_table(self, data_quality):
        """Add data overview table"""
        self.story.append(Paragraph("DATA OVERVIEW", self.heading_style))
        
        # Create overview table
        overview_data = [
            ['Metric', 'Value', 'Description'],
            ['Total Records', f"{data_quality['total_records']:,}", 'Number of data rows'],
            ['Total Columns', str(data_quality['total_columns']), 'Number of variables'],
            ['Numeric Columns', str(data_quality['numeric_columns']), 'Quantitative variables'],
            ['Categorical Columns', str(data_quality['categorical_columns']), 'Qualitative variables'],
            ['Missing Data', f"{data_quality['missing_percentage']}%", 'Percentage of missing values'],
            ['Duplicate Rows', str(data_quality['duplicate_rows']), 'Identical records found'],
            ['Memory Usage', f"{data_quality['memory_usage_mb']} MB", 'Dataset size in memory']
        ]
        
        table = Table(overview_data, colWidths=[2*inch, 1.5*inch, 2.5*inch])
        table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.darkblue),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 12),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('BACKGROUND', (0, 1), (-1, -1), colors.lightblue),
            ('GRID', (0, 0), (-1, -1), 1, colors.black),
            ('FONTSIZE', (0, 1), (-1, -1), 10),
        ]))
        
        self.story.append(table)
        self.story.append(Spacer(1, 20))
    
    def add_statistical_analysis(self, basic_stats):
        """Add detailed statistical analysis"""
        self.story.append(Paragraph("STATISTICAL ANALYSIS", self.heading_style))
        
        for column, stats in basic_stats.items():
            self.story.append(Paragraph(f"Analysis: {column}", self.subheading_style))
            
            # Create statistics table
            stats_data = [
                ['Statistic', 'Value'],
                ['Count', f"{stats['count']:,}"],
                ['Mean', f"{stats['mean']:,.3f}"],
                ['Median', f"{stats['median']:,.3f}"],
                ['Std Deviation', f"{stats['std']:,.3f}"],
                ['Minimum', f"{stats['min']:,.3f}"],
                ['Maximum', f"{stats['max']:,.3f}"],
                ['25th Percentile', f"{stats['q25']:,.3f}"],
                ['75th Percentile', f"{stats['q75']:,.3f}"],
                ['Missing Values', f"{stats['null_count']} ({stats['null_percentage']}%)"]
            ]
            
            table = Table(stats_data, colWidths=[2*inch, 2*inch])
            table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.darkgreen),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, 0), 11),
                ('BACKGROUND', (0, 1), (-1, -1), colors.lightgreen),
                ('GRID', (0, 0), (-1, -1), 1, colors.black),
                ('FONTSIZE', (0, 1), (-1, -1), 10),
            ]))
            
            self.story.append(table)
            self.story.append(Spacer(1, 15))
    
    def add_chart(self, chart_path, title, description=""):
        """Add chart to report"""
        if chart_path and os.path.exists(chart_path):
            try:
                self.story.append(Paragraph(title, self.subheading_style))
                img = Image(chart_path, width=6*inch, height=4.5*inch)
                self.story.append(img)
                if description:
                    self.story.append(Paragraph(description, self.body_style))
                self.story.append(Spacer(1, 20))
            except Exception as e:
                logger.error(f"Error adding chart {chart_path}: {e}")
    
    def add_insights(self, insights):
        """Add insights and recommendations"""
        self.story.append(Paragraph("KEY INSIGHTS & RECOMMENDATIONS", self.heading_style))
        
        for i, insight in enumerate(insights, 1):
            self.story.append(Paragraph(f"{i}. {insight}", self.body_style))
            self.story.append(Spacer(1, 8))
    
    def add_conclusion(self):
        """Add conclusion section"""
        self.story.append(Paragraph("CONCLUSION", self.heading_style))
        
        conclusion_text = """
        This automated analysis provides a comprehensive overview of the dataset's characteristics, 
        quality, and key patterns. The generated insights can guide strategic decisions and highlight 
        areas for further investigation. Regular automated reporting ensures consistent monitoring 
        of data trends and maintains data quality standards.
        
        For questions or additional analysis requirements, please contact the data analytics team.
        """
        
        self.story.append(Paragraph(conclusion_text, self.body_style))
        
        # Footer
        self.story.append(Spacer(1, 30))
        footer_text = f"Report generated by CODTECH Automated Report Generator on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
        footer_style = ParagraphStyle('Footer', parent=self.body_style, 
                                    fontSize=9, textColor=colors.grey, alignment=TA_CENTER)
        self.story.append(Paragraph(footer_text, footer_style))
    
    def build_report(self):
        """Build and save the PDF report"""
        try:
            doc = SimpleDocTemplate(self.filename, pagesize=A4,
                                  topMargin=72, bottomMargin=72,
                                  leftMargin=72, rightMargin=72)
            doc.build(self.story)
            logger.info(f"PDF report successfully generated: {self.filename}")
            return True
        except Exception as e:
            logger.error(f"Error building PDF report: {e}")
            return False

class AutomatedReportGenerator:
    """Main class orchestrating the report generation process"""
    
    def __init__(self, output_filename="Automated_Analysis_Report.pdf"):
        self.output_filename = output_filename
        self.data = None
        self.analyzer = None
        self.chart_generator = None
        self.pdf_generator = None
    
    def load_data(self, file_path):
        """Load data from various file formats"""
        if not os.path.exists(file_path):
            logger.error(f"File not found: {file_path}")
            return False
        
        try:
            file_extension = os.path.splitext(file_path)[1].lower()
            
            if file_extension == '.csv':
                self.data = pd.read_csv(file_path)
                logger.info(f"CSV file loaded successfully: {self.data.shape}")
                
            elif file_extension in ['.xlsx', '.xls']:
                self.data = pd.read_excel(file_path)
                logger.info(f"Excel file loaded successfully: {self.data.shape}")
                
            elif file_extension == '.json':
                self.data = pd.read_json(file_path)
                logger.info(f"JSON file loaded successfully: {self.data.shape}")
                
            else:
                logger.error(f"Unsupported file format: {file_extension}")
                return False
            
            # Basic data validation
            if self.data.empty:
                logger.error("Loaded data is empty")
                return False
                
            return True
            
        except Exception as e:
            logger.error(f"Error loading data: {e}")
            return False
    
    def generate_sample_data(self):
        """Generate comprehensive sample data for demonstration"""
        logger.info("Generating sample data for demonstration...")
        
        np.random.seed(42)
        n_records = 1000
        
        # Create realistic e-commerce dataset
        data = {
            'order_id': range(1, n_records + 1),
            'customer_id': np.random.randint(1000, 9999, n_records),
            'product_category': np.random.choice([
                'Electronics', 'Clothing', 'Home & Garden', 'Books', 
                'Sports & Outdoors', 'Beauty', 'Automotive'
            ], n_records, p=[0.25, 0.2, 0.15, 0.1, 0.12, 0.1, 0.08]),
            'product_price': np.round(np.random.lognormal(3.5, 1, n_records), 2),
            'quantity': np.random.poisson(2, n_records) + 1,
            'discount_percent': np.round(np.random.exponential(5, n_records), 1),
            'customer_age': np.random.normal(40, 15, n_records).astype(int),
            'customer_rating': np.round(np.random.beta(8, 2, n_records) * 4 + 1, 1),
            'shipping_cost': np.round(np.random.uniform(5, 25, n_records), 2),
            'delivery_days': np.random.poisson(5, n_records) + 1,
            'region': np.random.choice([
                'North America', 'Europe', 'Asia Pacific', 'South America', 'Africa'
            ], n_records, p=[0.35, 0.28, 0.22, 0.1, 0.05])
        }
        
        df = pd.DataFrame(data)
        
        # Calculate derived columns
        df['subtotal'] = df['product_price'] * df['quantity']
        df['discount_amount'] = df['subtotal'] * (df['discount_percent'] / 100)
        df['total_amount'] = df['subtotal'] - df['discount_amount'] + df['shipping_cost']
        
        # Apply realistic constraints
        df['customer_age'] = df['customer_age'].clip(18, 75)
        df['customer_rating'] = df['customer_rating'].clip(1, 5)
        df['delivery_days'] = df['delivery_days'].clip(1, 21)
        df['discount_percent'] = df['discount_percent'].clip(0, 50)
        
        # Add realistic missing values
        missing_indices = np.random.choice(df.index, size=int(0.03 * len(df)), replace=False)
        df.loc[missing_indices[:15], 'customer_rating'] = np.nan
        df.loc[missing_indices[15:25], 'discount_percent'] = np.nan
        df.loc[missing_indices[25:], 'delivery_days'] = np.nan
        
        self.data = df
        logger.info(f"Sample data generated: {self.data.shape}")
        return True
    
    def generate_insights(self, analysis_results):
        """Generate actionable insights from analysis results"""
        insights = []
        
        try:
            # Data quality insights
            data_quality = analysis_results.get('data_quality', {})
            missing_pct = data_quality.get('missing_percentage', 0)
            
            if missing_pct < 1:
                insights.append("Excellent data quality with minimal missing values detected.")
            elif missing_pct < 5:
                insights.append("Good data quality with acceptable levels of missing data.")
            else:
                insights.append(f"Data quality attention needed: {missing_pct}% missing values found.")
            
            if data_quality.get('duplicate_rows', 0) > 0:
                insights.append(f"Found {data_quality['duplicate_rows']} duplicate records requiring cleanup.")
            
            # Statistical insights
            basic_stats = analysis_results.get('basic_stats', {})
            for col, stats in basic_stats.items():
                cv = stats['std'] / stats['mean'] if stats['mean'] != 0 else 0
                if cv > 1:
                    insights.append(f"{col} shows high variability (CV > 1.0) - investigate outliers.")
                elif cv < 0.1:
                    insights.append(f"{col} shows low variability - data is highly consistent.")
            
            # Correlation insights
            correlation = analysis_results.get('correlation')
            if correlation is not None:
                # Find strong correlations
                strong_corr = []
                for i in range(len(correlation.columns)):
                    for j in range(i+1, len(correlation.columns)):
                        corr_val = correlation.iloc[i, j]
                        if abs(corr_val) > 0.7:
                            strong_corr.append((correlation.columns[i], correlation.columns[j], corr_val))
                
                if strong_corr:
                    for col1, col2, corr in strong_corr[:3]:  # Top 3
                        insights.append(f"Strong correlation ({corr:.2f}) between {col1} and {col2}.")
            
            # Categorical insights
            categorical = analysis_results.get('categorical', {})
            for col, stats in categorical.items():
                if stats['unique_count'] == 1:
                    insights.append(f"{col} has only one unique value - consider removing.")
                elif stats['unique_count'] > len(self.data) * 0.9:
                    insights.append(f"{col} has high cardinality - mostly unique values.")
            
            # General recommendations
            insights.extend([
                "Implement regular data quality monitoring for continuous improvement.",
                "Consider automated data validation rules to prevent future quality issues.",
                "Schedule periodic reports to track data trends and patterns over time."
            ])
            
        except Exception as e:
            logger.warning(f"Error generating insights: {e}")
            insights = [
                "Data analysis completed successfully.",
                "Consider domain-specific analysis for deeper insights.",
                "Regular monitoring recommended for data quality maintenance."
            ]
        
        return insights[:10]  # Limit to top 10 insights
    
    def generate_complete_report(self, file_path=None):
        """Generate the complete automated report"""
        logger.info("Starting automated report generation process...")
        
        # Load or generate data
        if file_path:
            success = self.load_data(file_path)
            if not success:
                logger.info("Failed to load file. Using sample data instead.")
                self.generate_sample_data()
        else:
            logger.info("No file provided. Generating sample data for demonstration.")
            self.generate_sample_data()
        
        # Initialize components
        self.analyzer = DataAnalyzer(self.data)
        self.chart_generator = ChartGenerator(self.data)
        self.pdf_generator = PDFReportGenerator(self.output_filename)
        
        # Perform comprehensive analysis
        logger.info("Performing data analysis...")
        basic_stats = self.analyzer.basic_statistics()
        correlation = self.analyzer.correlation_analysis()
        categorical = self.analyzer.categorical_analysis()
        data_quality = self.analyzer.data_quality_assessment()
        
        # Generate visualizations
        logger.info("Creating visualizations...")
        dist_chart = self.chart_generator.create_distribution_plots()
        corr_chart = self.chart_generator.create_correlation_heatmap(correlation)
        cat_chart = self.chart_generator.create_categorical_charts()
        
        # Generate insights
        insights = self.generate_insights(self.analyzer.analysis_results)
        
        # Build PDF report
        logger.info("Building PDF report...")
        
        # Add report sections
        self.pdf_generator.add_title_page()
        self.pdf_generator.add_executive_summary(data_quality)
        self.pdf_generator.add_data_overview_table(data_quality)
        
        if basic_stats:
            self.pdf_generator.add_statistical_analysis(basic_stats)
        
        # Add charts with descriptions
        if dist_chart:
            self.pdf_generator.add_chart(dist_chart, "Data Distribution Analysis", 
                                       "Histograms showing the distribution patterns of numeric variables.")
        
        if corr_chart:
            self.pdf_generator.add_chart(corr_chart, "Correlation Analysis", 
                                       "Heatmap displaying relationships between numeric variables.")
        
        if cat_chart:
            self.pdf_generator.add_chart(cat_chart, "Categorical Data Analysis", 
                                       "Bar charts showing the frequency distribution of categorical variables.")
        
        # Add insights and conclusion
        self.pdf_generator.add_insights(insights)
        self.pdf_generator.add_conclusion()
        
        # Generate final PDF
        success = self.pdf_generator.build_report()
        
        # Cleanup
        self.chart_generator.cleanup_charts()
        
        if success:
            logger.info(f"Report generation completed successfully: {self.output_filename}")
            return self.output_filename
        else:
            logger.error("Report generation failed")
            return None
    
    def print_summary(self):
        """Print analysis summary to console"""
        if not self.analyzer or not self.analyzer.analysis_results:
            return
        
        print("\n" + "="*60)
        print("AUTOMATED REPORT GENERATION SUMMARY")
        print("="*60)
        
        data_quality = self.analyzer.analysis_results.get('data_quality', {})
        print(f"📊 Dataset Size: {data_quality.get('total_records', 0):,} rows × {data_quality.get('total_columns', 0)} columns")
        print(f"🔢 Numeric Variables: {data_quality.get('numeric_columns', 0)}")
        print(f"📝 Categorical Variables: {data_quality.get('categorical_columns', 0)}")
        print(f"❌ Missing Data: {data_quality.get('missing_percentage', 0)}%")
        print(f"🔁 Duplicate Rows: {data_quality.get('duplicate_rows', 0)}")
        print(f"💾 Memory Usage: {data_quality.get('memory_usage_mb', 0)} MB")
        
        print(f"\n✅ Report Generated: {self.output_filename}")
        print("📈 Includes: Statistical Analysis, Visualizations, Data Quality Assessment")
        print("🎯 Ready for: Decision Making, Further Analysis, Presentation")
        print("="*60)

def create_sample_file():
    """Create a sample CSV file for testing"""
    try:
        # Generate sample data and save as CSV
        generator = AutomatedReportGenerator()
        generator.generate_sample_data()
        
        sample_filename = "sample_ecommerce_data.csv"
        generator.data.to_csv(sample_filename, index=False)
        
        print(f"✅ Sample file created: {sample_filename}")
        print(f"📊 Contains: {len(generator.data)} records with realistic e-commerce data")
        return sample_filename
        
    except Exception as e:
        logger.error(f"Error creating sample file: {e}")
        return None

def main():
    """Main function to run the automated report generator"""
    print("="*60)
    print("CODTECH INTERNSHIP - TASK 2")
    print("AUTOMATED REPORT GENERATION SYSTEM")
    print("="*60)
    print("Features:")
    print("• Multi-format data support (CSV, Excel, JSON)")
    print("• Comprehensive statistical analysis")
    print("• Professional PDF report generation")
    print("• Data quality assessment")
    print("• Automated insights & recommendations")
    print("="*60)
    
    # Get user input
    print("\nOptions:")
    print("1. Analyze your own data file")
    print("2. Generate demo report with sample data")
    print("3. Create sample data file for testing")
    
    try:
        choice = input("\nEnter your choice (1-3): ").strip()
        
        if choice == "1":
            file_path = input("Enter the path to your data file: ").strip()
            if not file_path:
                print("❌ No file path provided. Using sample data instead.")
                file_path = None
                
        elif choice == "2":
            file_path = None
            print("📊 Using sample data for demonstration...")
            
        elif choice == "3":
            sample_file = create_sample_file()
            if sample_file:
                use_sample = input(f"\n🤔 Use the created sample file '{sample_file}' for analysis? (y/n): ").strip()
                if use_sample.lower() in ['y', 'yes']:
                    file_path = sample_file
                else:
                    print("👋 Sample file created. Run the script again to analyze it.")
                    return
            else:
                print("❌ Failed to create sample file.")
                return
        else:
            print("❌ Invalid choice. Using sample data.")
            file_path = None
        
        # Generate report
        output_filename = input("\nEnter output filename (press Enter for default): ").strip()
        if not output_filename:
            output_filename = "Automated_Analysis_Report.pdf"
        
        # Initialize and run generator
        generator = AutomatedReportGenerator(output_filename)
        
        print(f"\n🚀 Starting report generation...")
        result = generator.generate_complete_report(file_path)
        
        if result:
            generator.print_summary()
            print(f"\n🎉 SUCCESS! Open '{result}' to view your automated report.")
            
            # Ask if user wants to see data preview
            if generator.data is not None:
                show_preview = input("\n🔍 Show data preview? (y/n): ").strip()
                if show_preview.lower() in ['y', 'yes']:
                    print("\n" + "="*50)
                    print("DATA PREVIEW")
                    print("="*50)
                    print(generator.data.head(10))
                    print(f"\n... and {len(generator.data)-10} more rows")
        else:
            print("❌ Report generation failed. Check the logs for details.")
            
    except KeyboardInterrupt:
        print("\n\n👋 Process interrupted by user. Goodbye!")
    except Exception as e:
        logger.error(f"Unexpected error in main: {e}")
        print(f"❌ An unexpected error occurred: {e}")

if __name__ == "__main__":
    main()
