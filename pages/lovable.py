"""
Lovable - AI-Powered Application Builder
Build production apps with natural language
Built for Lovable by Anju Nandhakumar
"""

import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime
import plotly.graph_objects as go
from utils.sidebar import render_sidebar

render_sidebar()

# Page config
st.set_page_config(page_title="Lovable", page_icon="💜", layout="wide")

# App templates
APP_TEMPLATES = {
    'E-commerce Store': {
        'components': ['Product catalog', 'Shopping cart', 'Checkout', 'User auth', 'Admin dashboard'],
        'tech_stack': 'React + Node.js + PostgreSQL',
        'build_time': '8 minutes'
    },
    'SaaS Dashboard': {
        'components': ['Analytics', 'User management', 'Billing', 'Settings', 'API docs'],
        'tech_stack': 'Next.js + Supabase + Stripe',
        'build_time': '6 minutes'
    },
    'Social Platform': {
        'components': ['User profiles', 'Feed', 'Messaging', 'Notifications', 'Content moderation'],
        'tech_stack': 'React + Firebase + Cloudflare',
        'build_time': '10 minutes'
    },
    'Internal Tool': {
        'components': ['Data tables', 'Forms', 'Workflows', 'Reporting', 'Permissions'],
        'tech_stack': 'React + Python + PostgreSQL',
        'build_time': '5 minutes'
    }
}

# Code generation quality metrics
CODE_QUALITY = {
    'Type Safety': 98.5,
    'Best Practices': 96.2,
    'Performance': 94.8,
    'Security': 97.3,
    'Maintainability': 95.6,
    'Test Coverage': 88.9
}

# Header
st.markdown("""
<div style="text-align: center; padding: 40px 20px; background: linear-gradient(135deg, #7c3aed 0%, #73BA9B 100%); border-radius: 20px; margin-bottom: 30px; box-shadow: 0 8px 20px rgba(0,0,0,0.15);">
    <div style="display: inline-flex; align-items: center; gap: 20px; margin-bottom: 16px;">
        <div style="width: 70px; height: 70px; background: rgba(255,255,255,0.2); border-radius: 50%; display: flex; align-items: center; justify-content: center;">
            <span style="font-size: 40px;">💜</span>
        </div>
        <h1 style="font-size: 52px; font-weight: 900; color: white; margin: 0;">Lovable</h1>
    </div>
    <p style="font-size: 24px; color: white; font-weight: 700; margin: 12px 0;">AI-Powered Application Builder</p>
    <p style="font-size: 16px; color: rgba(255,255,255,0.9); font-weight: 500;">Build production apps with natural language • Ship in minutes, not months</p>
</div>
""", unsafe_allow_html=True)

# Tabs
tab1, tab2, tab3, tab4 = st.tabs(["🚀 Build App", "📊 Generated Code", "⚡ Performance", "💡 Platform Features"])

with tab1:
    st.markdown("### Build Your Application")
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.markdown("**Describe Your App**")
        
        app_description = st.text_area(
            "What do you want to build?",
            "Build me a project management tool with task boards, team collaboration, and time tracking. Include user authentication and real-time updates.",
            height=120
        )
        
        st.markdown("**Configuration**")
        
        template = st.selectbox("Start from template (optional)", ["None"] + list(APP_TEMPLATES.keys()))
        
        features = st.multiselect(
            "Required Features",
            ["User Authentication", "Database", "API Integration", "Real-time Updates", "File Uploads", "Email Notifications", "Payment Processing"],
            default=["User Authentication", "Database", "Real-time Updates"]
        )
        
        tech_stack = st.selectbox(
            "Tech Stack Preference",
            ["Auto-select (Recommended)", "React + Node.js", "Next.js + Supabase", "Vue + Python", "Svelte + Go"]
        )
        
        build_btn = st.button("💜 Generate Application", type="primary", use_container_width=True)
    
    with col2:
        if build_btn:
            st.markdown("**Generation Progress**")
            
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            import time
            stages = [
                ("Analyzing requirements...", 0.15),
                ("Designing architecture...", 0.3),
                ("Generating frontend components...", 0.5),
                ("Building backend APIs...", 0.65),
                ("Setting up database...", 0.8),
                ("Configuring deployment...", 0.95),
                ("Running tests...", 1.0)
            ]
            
            for stage, progress in stages:
                status_text.text(stage)
                progress_bar.progress(progress)
                time.sleep(0.5)
            
            st.success("✅ Application generated successfully!")
            
            st.markdown("""
            <div style="background: linear-gradient(135deg, #7c3aed 0%, #73BA9B 100%); padding: 25px; border-radius: 16px; margin: 20px 0; box-shadow: 0 8px 20px rgba(0,0,0,0.15);">
                <h3 style="color: white; margin: 0 0 20px 0; font-size: 24px; font-weight: 900;">Application Generated</h3>
                <div style="display: grid; grid-template-columns: repeat(2, 1fr); gap: 15px;">
                    <div style="background: rgba(255,255,255,0.15); backdrop-filter: blur(10px); border-radius: 12px; padding: 18px;">
                        <p style="font-size: 14px; color: rgba(255,255,255,0.8); margin: 0 0 8px 0;">Components</p>
                        <p style="font-size: 18px; color: white; font-weight: 700; margin: 0;">24 files</p>
                    </div>
                    <div style="background: rgba(255,255,255,0.15); backdrop-filter: blur(10px); border-radius: 12px; padding: 18px;">
                        <p style="font-size: 14px; color: rgba(255,255,255,0.8); margin: 0 0 8px 0;">Lines of Code</p>
                        <p style="font-size: 18px; color: white; font-weight: 700; margin: 0;">3,847</p>
                    </div>
                    <div style="background: rgba(255,255,255,0.15); backdrop-filter: blur(10px); border-radius: 12px; padding: 18px;">
                        <p style="font-size: 14px; color: rgba(255,255,255,0.8); margin: 0 0 8px 0;">Build Time</p>
                        <p style="font-size: 18px; color: white; font-weight: 700; margin: 0;">6.2 minutes</p>
                    </div>
                    <div style="background: rgba(255,255,255,0.15); backdrop-filter: blur(10px); border-radius: 12px; padding: 18px;">
                        <p style="font-size: 14px; color: rgba(255,255,255,0.8); margin: 0 0 8px 0;">Deploy URL</p>
                        <p style="font-size: 18px; color: white; font-weight: 700; margin: 0;">app.lovable.dev/...</p>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Type Safety", "98.5%", "✓")
            col2.metric("Test Coverage", "88.9%", "✓")
            col3.metric("Performance", "94.8/100", "A")
            col4.metric("Security", "97.3/100", "A+")

with tab2:
    st.markdown("### Generated Code Quality")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown("**Sample Generated Component**")
        
        st.code("""
// TaskBoard.tsx - Generated by Lovable AI
import React, { useState, useEffect } from 'react';
import { DragDropContext, Droppable, Draggable } from 'react-beautiful-dnd';
import { Task, TaskStatus } from '@/types';
import { useTaskStore } from '@/store/tasks';
import { TaskCard } from './TaskCard';

export const TaskBoard: React.FC = () => {
  const { tasks, updateTaskStatus } = useTaskStore();
  const [columns, setColumns] = useState<Record<TaskStatus, Task[]>>({
    todo: [],
    inProgress: [],
    done: []
  });

  useEffect(() => {
    const grouped = tasks.reduce((acc, task) => {
      acc[task.status] = [...(acc[task.status] || []), task];
      return acc;
    }, {} as Record<TaskStatus, Task[]>);
    
    setColumns(grouped);
  }, [tasks]);

  const handleDragEnd = (result: any) => {
    if (!result.destination) return;
    
    const { source, destination, draggableId } = result;
    
    if (source.droppableId !== destination.droppableId) {
      updateTaskStatus(draggableId, destination.droppableId as TaskStatus);
    }
  };

  return (
    <DragDropContext onDragEnd={handleDragEnd}>
      <div className="grid grid-cols-3 gap-4">
        {Object.entries(columns).map(([status, tasks]) => (
          <Droppable key={status} droppableId={status}>
            {(provided) => (
              <div
                ref={provided.innerRef}
                {...provided.droppableProps}
                className="bg-gray-50 rounded-lg p-4"
              >
                <h3 className="font-semibold mb-4">{status}</h3>
                {tasks.map((task, index) => (
                  <Draggable key={task.id} draggableId={task.id} index={index}>
                    {(provided) => (
                      <div
                        ref={provided.innerRef}
                        {...provided.draggableProps}
                        {...provided.dragHandleProps}
                      >
                        <TaskCard task={task} />
                      </div>
                    )}
                  </Draggable>
                ))}
                {provided.placeholder}
              </div>
            )}
          </Droppable>
        ))}
      </div>
    </DragDropContext>
  );
};
""", language="typescript")
    
    with col2:
        st.markdown("**Code Quality Metrics**")
        
        quality_data = []
        for metric, score in CODE_QUALITY.items():
            grade = 'A+' if score >= 97 else 'A' if score >= 93 else 'B+' if score >= 88 else 'B'
            quality_data.append({
                'Metric': metric,
                'Score': f"{score}%",
                'Grade': grade
            })
        
        st.dataframe(pd.DataFrame(quality_data), hide_index=True, use_container_width=True)
        
        st.markdown("**Code Quality Distribution**")
        
        fig1 = go.Figure(data=[go.Bar(
            x=list(CODE_QUALITY.keys()),
            y=list(CODE_QUALITY.values()),
            marker=dict(color='#7c3aed'),
            text=[f"{v}%" for v in CODE_QUALITY.values()],
            textposition='auto'
        )])
        fig1.update_layout(
            yaxis=dict(range=[80, 100]),
            height=250
        )
        st.plotly_chart(fig1, use_container_width=True)

with tab3:
    st.markdown("### Development Speed Comparison")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown("**Time to Production**")
        
        comparison = {
            'Method': ['Lovable AI', 'No-Code Platform', 'Traditional Dev (Junior)', 'Traditional Dev (Senior)', 'Outsourced Agency'],
            'Build Time': ['6 minutes', '2 days', '2 weeks', '1 week', '4 weeks'],
            'Cost': ['$0', '$500', '$8,000', '$12,000', '$25,000'],
            'Code Quality': ['A+', 'N/A', 'B', 'A', 'A']
        }
        st.dataframe(pd.DataFrame(comparison), hide_index=True, use_container_width=True)
        
        st.markdown("**💰 Lovable ROI: Save $8K-$25K per project**")
    
    with col2:
        st.markdown("**Build Time Comparison**")
        
        # Convert to minutes for visualization
        times = {
            'Lovable AI': 6,
            'No-Code': 2880,  # 2 days
            'Junior Dev': 20160,  # 2 weeks
            'Senior Dev': 10080,  # 1 week
            'Agency': 40320  # 4 weeks
        }
        
        fig2 = go.Figure(data=[go.Bar(
            y=list(times.keys()),
            x=list(times.values()),
            orientation='h',
            marker=dict(color=['#10b981', '#f59e0b', '#ef4444', '#ef4444', '#ef4444']),
            text=[f"{v} min" if v < 100 else f"{v//1440} days" for v in times.values()],
            textposition='auto'
        )])
        fig2.update_layout(
            xaxis_type="log",
            xaxis_title='Time (minutes, log scale)',
            height=300
        )
        st.plotly_chart(fig2, use_container_width=True)
    
    st.markdown("**Performance Benchmarks**")
    
    col1, col2, col3 = st.columns(3)
    col1.metric("Apps Built", "12,847", "+2,341 this month")
    col2.metric("Avg Lines of Code", "3,200", "per app")
    col3.metric("Success Rate", "96.8%", "+1.2%")

with tab4:
    st.markdown("### Platform Capabilities")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**AI Code Generation**")
        st.markdown("""
        - ✅ Natural language to production code
        - ✅ Full-stack applications (frontend + backend)
        - ✅ Database schema generation
        - ✅ API endpoint creation
        - ✅ Authentication & authorization
        - ✅ Real-time features (WebSocket)
        - ✅ Payment integration (Stripe)
        - ✅ Deployment configuration
        """)
        
        st.markdown("**Tech Stack Support**")
        st.markdown("""
        - ✅ React, Next.js, Vue, Svelte
        - ✅ Node.js, Python, Go
        - ✅ PostgreSQL, MongoDB, Supabase
        - ✅ TypeScript by default
        - ✅ Tailwind CSS styling
        - ✅ Jest/Vitest testing
        """)
    
    with col2:
        st.markdown("**Code Quality**")
        st.markdown("""
        - ✅ 98.5% type safety (TypeScript)
        - ✅ 88.9% test coverage
        - ✅ ESLint + Prettier formatting
        - ✅ Security best practices
        - ✅ Performance optimization
        - ✅ Accessibility (WCAG 2.1)
        """)
        
        st.markdown("**Deployment & DevOps**")
        st.markdown("""
        - ✅ One-click deployment
        - ✅ Auto-scaling infrastructure
        - ✅ CI/CD pipelines
        - ✅ Environment management
        - ✅ Monitoring & logging
        - ✅ Rollback capability
        """)
    
    st.markdown("**Generated App Components**")
    
    components = {
        'Category': ['UI Components', 'Backend APIs', 'Database', 'Auth', 'Testing', 'Config'],
        'Files Generated': [12, 6, 3, 2, 8, 3],
        'Lines of Code': [1840, 920, 380, 280, 320, 107]
    }
    st.dataframe(pd.DataFrame(components), hide_index=True, use_container_width=True)

# Features
st.markdown("""
<div style="margin-top: 40px; padding: 30px; background: linear-gradient(135deg, #f5f3ff 0%, #ede9fe 100%); border-radius: 16px;">
    <h3 style="margin: 0 0 20px 0; color: #5b21b6; font-size: 24px; font-weight: 800;">💡 Platform Features</h3>
    <div style="display: grid; grid-template-columns: repeat(2, 1fr); gap: 15px;">
        <div style="background: white; border-radius: 12px; padding: 18px;">
            <p style="font-size: 14px; color: #7c3aed; font-weight: 700; margin: 0 0 6px 0;">✓ Natural Language</p>
            <p style="font-size: 13px; color: #6b7280; margin: 0;">Describe app, get production code</p>
        </div>
        <div style="background: white; border-radius: 12px; padding: 18px;">
            <p style="font-size: 14px; color: #7c3aed; font-weight: 700; margin: 0 0 6px 0;">✓ 6 Minute Build</p>
            <p style="font-size: 13px; color: #6b7280; margin: 0;">Ship apps in minutes, not months</p>
        </div>
        <div style="background: white; border-radius: 12px; padding: 18px;">
            <p style="font-size: 14px; color: #7c3aed; font-weight: 700; margin: 0 0 6px 0;">✓ Production Quality</p>
            <p style="font-size: 13px; color: #6b7280; margin: 0;">98.5% type safety, 88.9% coverage</p>
        </div>
        <div style="background: white; border-radius: 12px; padding: 18px;">
            <p style="font-size: 14px; color: #7c3aed; font-weight: 700; margin: 0 0 6px 0;">✓ Full Stack</p>
            <p style="font-size: 13px; color: #6b7280; margin: 0;">Frontend, backend, database, deploy</p>
        </div>
    </div>
</div>
<div style="text-align: center; padding: 30px; margin-top: 20px; background: linear-gradient(135deg, #7c3aed 0%, #73BA9B 100%); border-radius: 16px; color: white;">
    <h3 style="margin: 0 0 15px 0; font-size: 24px; font-weight: 900;">Built for Lovable</h3>
    <p style="font-size: 16px; margin: 8px 0; font-weight: 600;">Anju Vilashni Nandhakumar • MS AI @ Northeastern (2025)</p>
    <p style="font-size: 14px; margin: 8px 0;">📧 nandhakumar.anju@gmail.com • 🔗 <a href="https://vxanju.com" style="color: white;">vxanju.com</a></p>
</div>
""", unsafe_allow_html=True)