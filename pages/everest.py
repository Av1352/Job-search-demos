"""
Everest - AI IT Support Agent
Automated ticket resolution and IT operations
Built for Everest by Anju Nandhakumar
"""

import streamlit as st
import pandas as pd
from datetime import datetime, timedelta
import random
from utils.sidebar import render_sidebar
render_sidebar()
# Page config
st.set_page_config(page_title="Everest - AI IT Support", layout="wide")

# Initialize session state
if 'ticket_analyzed' not in st.session_state:
    st.session_state.ticket_analyzed = False
if 'resolution_generated' not in st.session_state:
    st.session_state.resolution_generated = False

# Sample tickets for testing
SAMPLE_TICKETS = {
    "Password Reset Request": {
        "description": "I forgot my password for the company portal. Can someone help me reset it?",
        "user": "john.smith@company.com",
        "priority": "Medium",
        "category": "Account Access"
    },
    "VPN Connection Issue": {
        "description": "I can't connect to the VPN from home. Getting 'authentication failed' error. I've tried restarting my laptop.",
        "user": "sarah.jones@company.com",
        "priority": "High",
        "category": "Network"
    },
    "Software Installation": {
        "description": "Need Slack installed on my new laptop. Also need access to the engineering shared drive.",
        "user": "mike.chen@company.com",
        "priority": "Low",
        "category": "Software"
    },
    "Email Not Working": {
        "description": "My Outlook keeps crashing when I try to open it. Started this morning after the update.",
        "user": "lisa.park@company.com",
        "priority": "High",
        "category": "Email"
    }
}

def classify_ticket(description):
    """Classify ticket and determine if auto-resolvable"""
    description_lower = description.lower()
    
    # Password reset
    if 'password' in description_lower and 'reset' in description_lower:
        return {
            'category': 'Account Access',
            'subcategory': 'Password Reset',
            'priority': 'Medium',
            'auto_resolvable': True,
            'estimated_time': '2 minutes',
            'confidence': 0.95,
            'similar_tickets': 847,
            'avg_resolution_time': '3 minutes'
        }
    
    # VPN issues
    elif 'vpn' in description_lower or 'remote access' in description_lower:
        return {
            'category': 'Network',
            'subcategory': 'VPN Connection',
            'priority': 'High',
            'auto_resolvable': True,
            'estimated_time': '5 minutes',
            'confidence': 0.92,
            'similar_tickets': 623,
            'avg_resolution_time': '8 minutes'
        }
    
    # Software installation
    elif 'install' in description_lower or 'software' in description_lower:
        return {
            'category': 'Software',
            'subcategory': 'Application Installation',
            'priority': 'Low',
            'auto_resolvable': True,
            'estimated_time': '10 minutes',
            'confidence': 0.88,
            'similar_tickets': 512,
            'avg_resolution_time': '12 minutes'
        }
    
    # Email issues
    elif 'email' in description_lower or 'outlook' in description_lower:
        return {
            'category': 'Email',
            'subcategory': 'Client Issues',
            'priority': 'High',
            'auto_resolvable': True,
            'estimated_time': '8 minutes',
            'confidence': 0.90,
            'similar_tickets': 394,
            'avg_resolution_time': '10 minutes'
        }
    
    # Default
    else:
        return {
            'category': 'General IT',
            'subcategory': 'Uncategorized',
            'priority': 'Medium',
            'auto_resolvable': False,
            'estimated_time': 'Needs review',
            'confidence': 0.65,
            'similar_tickets': 0,
            'avg_resolution_time': 'N/A'
        }

def generate_resolution(classification):
    """Generate resolution steps based on ticket type"""
    category = classification['subcategory']
    
    if category == 'Password Reset':
        return {
            'resolution_type': 'Automated',
            'steps': [
                '✅ Password reset link sent to john.smith@company.com',
                '✅ Verified user identity via SSO',
                '✅ Generated temporary password valid for 24 hours',
                '✅ User will be prompted to set new password on next login',
                '✅ 2FA re-enrollment required for security'
            ],
            'status': 'Resolved',
            'time_to_resolve': '2 minutes',
            'user_action': 'Check your email for password reset link'
        }
    
    elif category == 'VPN Connection':
        return {
            'resolution_type': 'Automated',
            'steps': [
                '✅ Detected outdated VPN client version (v2.3)',
                '✅ Pushed latest VPN client update (v2.8) to device',
                '✅ Refreshed authentication certificates',
                '✅ Cleared local VPN cache',
                '✅ Verified network connectivity and firewall rules'
            ],
            'status': 'Resolved',
            'time_to_resolve': '5 minutes',
            'user_action': 'Restart VPN client and try connecting again'
        }
    
    elif category == 'Application Installation':
        return {
            'resolution_type': 'Automated',
            'steps': [
                '✅ Verified user has software license available',
                '✅ Deployed Slack v4.35.0 via company software center',
                '✅ Granted access to Engineering shared drive (\\\\fileserver\\engineering)',
                '✅ Added user to "Engineering Team" security group',
                '✅ Syncing shared drive to laptop (est. 10 min)'
            ],
            'status': 'Resolved',
            'time_to_resolve': '10 minutes',
            'user_action': 'Check your Applications folder - Slack should appear shortly'
        }
    
    elif category == 'Client Issues':
        return {
            'resolution_type': 'Automated',
            'steps': [
                '✅ Identified corrupted Outlook profile after recent update',
                '✅ Created backup of current profile and PST files',
                '✅ Rebuilt Outlook profile with clean configuration',
                '✅ Re-synced mailbox from Exchange server',
                '✅ Restored calendar, contacts, and folder structure'
            ],
            'status': 'Resolved',
            'time_to_resolve': '8 minutes',
            'user_action': 'Restart Outlook - your emails and calendar are restored'
        }
    
    else:
        return {
            'resolution_type': 'Escalated',
            'steps': [
                '⚠️ Issue requires human review',
                '📋 Ticket escalated to L2 support',
                '🔔 Engineer will be assigned within 15 minutes'
            ],
            'status': 'Escalated',
            'time_to_resolve': 'Pending',
            'user_action': 'An IT engineer will contact you shortly'
        }

# Header
st.markdown("""
    <div style="text-align: center; padding: 50px 30px; background: linear-gradient(135deg, #1e3a8a 0%, #3b82f6 100%); border-radius: 25px; margin-bottom: 35px; box-shadow: 0 12px 28px rgba(59, 130, 246, 0.35);">
        <div style="width: 100px; height: 100px; background: linear-gradient(135deg, #3b82f6 0%, #2563eb 100%); border-radius: 50%; display: flex; align-items: center; justify-content: center; box-shadow: 0 8px 20px rgba(59, 130, 246, 0.5); margin: 0 auto 25px auto; border: 5px solid white;">
            <span style="font-size: 56px;">🤖</span>
        </div>
        <h1 style="font-size: 58px; font-weight: 900; color: white; margin: 0 0 18px 0; text-shadow: 0 4px 8px rgba(0,0,0,0.2);">
            Everest
        </h1>
        <p style="font-size: 28px; color: rgba(255,255,255,0.95); font-weight: 700; margin: 15px 0;">AI IT Support Agent</p>
        <p style="font-size: 18px; color: rgba(255,255,255,0.85); font-weight: 500; margin-bottom: 25px;">Automated ticket resolution and IT operations</p>
        <div style="display: flex; gap: 14px; flex-wrap: wrap; justify-content: center; align-items: center; max-width: 850px; margin: 28px auto 0 auto;">
            <span style="background: linear-gradient(135deg, #ec4899 0%, #f43f5e 100%); color: white; padding: 10px 22px; border-radius: 30px; font-size: 15px; font-weight: 800; box-shadow: 0 4px 12px rgba(236, 72, 153, 0.4);">Ticket Automation</span>
            <span style="background: linear-gradient(135deg, #8b5cf6 0%, #7c3aed 100%); color: white; padding: 10px 22px; border-radius: 30px; font-size: 15px; font-weight: 800; box-shadow: 0 4px 12px rgba(139, 92, 246, 0.4);">Smart Routing</span>
            <span style="background: linear-gradient(135deg, #f59e0b 0%, #ea580c 100%); color: white; padding: 10px 22px; border-radius: 30px; font-size: 15px; font-weight: 800; box-shadow: 0 4px 12px rgba(245, 158, 11, 0.4);">Auto-Resolution</span>
            <span style="background: linear-gradient(135deg, #10b981 0%, #059669 100%); color: white; padding: 10px 22px; border-radius: 30px; font-size: 15px; font-weight: 800; box-shadow: 0 4px 12px rgba(16, 185, 129, 0.4);">YC Backed</span>
        </div>
        <p style="font-size: 16px; color: rgba(255,255,255,0.9); margin-top: 25px; font-weight: 600;">
            Built for <strong style="color: white;">Everest</strong> by <strong style="color: white;">Anju Nandhakumar</strong>
        </p>
    </div>
    """, unsafe_allow_html=True)

# Value Prop
st.markdown("""
<div style="background: linear-gradient(135deg, #ecfdf5, #d1fae5); padding: 25px; border-radius: 15px; border: 2px solid #059669; margin-bottom: 30px;">
    <h3 style="color: #065f46; margin: 0 0 15px 0; font-size: 22px; font-weight: 800;">🎯 The IT Support Problem</h3>
    <div style="display: grid; grid-template-columns: repeat(3, 1fr); gap: 15px;">
        <div style="background: white; padding: 18px; border-radius: 10px;">
            <p style="color: #ef4444; font-weight: 700; margin: 0 0 8px 0;">❌ Today</p>
            <p style="color: #6b7280; font-size: 14px; margin: 0;">Employees wait 4+ hours for simple fixes. 70% of tickets are routine issues. IT teams overwhelmed.</p>
        </div>
        <div style="background: white; padding: 18px; border-radius: 10px;">
            <p style="color: #f59e0b; font-weight: 700; margin: 0 0 8px 0;">💰 Cost Impact</p>
            <p style="color: #6b7280; font-size: 14px; margin: 0;">$52/ticket average cost. Companies spend $150K/year on outsourced IT for 500 employees.</p>
        </div>
        <div style="background: white; padding: 18px; border-radius: 10px;">
            <p style="color: #10b981; font-weight: 700; margin: 0 0 8px 0;">✅ With Everest</p>
            <p style="color: #6b7280; font-size: 14px; margin: 0;">Resolve 70% of tickets in <5 min. Save $105K/year. Zero wait time for employees.</p>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

# Tabs
tab1, tab2, tab3 = st.tabs(["🎫 Submit Ticket", "📊 Team Dashboard", "🔧 How It Works"])

with tab1:
    st.markdown("""
    <div style="background: linear-gradient(135deg, #f0f9ff 0%, #e0f2fe 100%); border: 3px solid #3b82f6; border-radius: 16px; padding: 24px; margin-bottom: 20px;">
        <h3 style="color: #1e40af; font-size: 22px; font-weight: 800; margin: 0;">Need IT Help?</h3>
        <p style="color: #3b82f6; font-size: 14px; margin: 8px 0 0 0;">Tell us what's wrong - most issues resolve automatically in under 5 minutes</p>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns([3, 2])
    
    with col1:
        # Ticket selection
        use_sample = st.checkbox("Use sample ticket", value=True)
        
        if use_sample:
            ticket_name = st.selectbox("Select sample issue", list(SAMPLE_TICKETS.keys()))
            ticket = SAMPLE_TICKETS[ticket_name]
            issue_desc = ticket['description']
        else:
            issue_desc = st.text_area(
                "Describe your issue",
                placeholder="Example: I can't connect to WiFi in the office. My laptop shows 'Limited connectivity'.",
                height=100
            )
        
        if issue_desc:
            st.text_area("Issue Description", issue_desc, height=100, disabled=True)
            
            if st.button("🚀 Submit Ticket", type="primary", use_container_width=True):
                st.session_state.ticket_analyzed = True
                st.session_state.resolution_generated = True
    
    with col2:
        st.markdown("""
        <div style="background: #fef3c7; padding: 20px; border-radius: 12px; border-left: 4px solid #f59e0b;">
            <h4 style="color: #92400e; margin: 0 0 12px 0; font-size: 16px;">⚡ Fast Resolution</h4>
            <ul style="color: #78350f; font-size: 13px; line-height: 1.8; margin: 0; padding-left: 20px;">
                <li>70% of tickets auto-resolve in <5 min</li>
                <li>No waiting in queue</li>
                <li>24/7 availability</li>
                <li>Instant password resets</li>
                <li>Automatic software deployment</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    if st.session_state.ticket_analyzed:
        st.markdown("<hr style='margin: 30px 0; border: 1px solid #e5e7eb;'>", unsafe_allow_html=True)
        
        # Classify ticket
        classification = classify_ticket(issue_desc)
        
        # Show classification
        st.markdown(f"""
        <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); padding: 30px; border-radius: 20px; margin-bottom: 25px;">
            <h2 style="color: white; font-size: 28px; font-weight: 900; margin: 0 0 20px 0;">🎯 Ticket Analysis</h2>
            <div style="display: grid; grid-template-columns: repeat(4, 1fr); gap: 15px;">
                <div style="background: rgba(255,255,255,0.15); backdrop-filter: blur(10px); border-radius: 12px; padding: 20px; text-align: center;">
                    <p style="font-size: 14px; color: rgba(255,255,255,0.8); margin: 0;">Category</p>
                    <p style="font-size: 20px; color: white; font-weight: 900; margin: 8px 0;">{classification['category']}</p>
                    <p style="font-size: 13px; color: rgba(255,255,255,0.7); margin: 0;">{classification['subcategory']}</p>
                </div>
                <div style="background: rgba(255,255,255,0.15); backdrop-filter: blur(10px); border-radius: 12px; padding: 20px; text-align: center;">
                    <p style="font-size: 14px; color: rgba(255,255,255,0.8); margin: 0;">Priority</p>
                    <p style="font-size: 32px; color: white; font-weight: 900; margin: 8px 0;">{classification['priority']}</p>
                </div>
                <div style="background: rgba(255,255,255,0.15); backdrop-filter: blur(10px); border-radius: 12px; padding: 20px; text-align: center;">
                    <p style="font-size: 14px; color: rgba(255,255,255,0.8); margin: 0;">Confidence</p>
                    <p style="font-size: 32px; color: #86efac; font-weight: 900; margin: 8px 0;">{classification['confidence']:.0%}</p>
                </div>
                <div style="background: rgba(255,255,255,0.15); backdrop-filter: blur(10px); border-radius: 12px; padding: 20px; text-align: center;">
                    <p style="font-size: 14px; color: rgba(255,255,255,0.8); margin: 0;">Est. Time</p>
                    <p style="font-size: 28px; color: white; font-weight: 900; margin: 8px 0;">{classification['estimated_time']}</p>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        if classification['auto_resolvable']:
            # Generate and show resolution
            resolution = generate_resolution(classification)
            
            st.success(f"✅ **Ticket Status:** {resolution['status']} in {resolution['time_to_resolve']}")
            
            col_x, col_y = st.columns([2, 1])
            
            with col_x:
                st.markdown(f"""
                <div style="background: #ecfdf5; padding: 25px; border-radius: 15px; border: 2px solid #059669;">
                    <h3 style="color: #065f46; margin: 0 0 15px 0; font-size: 20px;">🔧 Resolution Steps</h3>
                    <div style="background: white; padding: 20px; border-radius: 10px;">
                        {''.join([f'<p style="color: #047857; font-size: 14px; line-height: 1.8; margin: 8px 0;">{step}</p>' for step in resolution['steps']])}
                    </div>
                </div>
                """, unsafe_allow_html=True)
            
            with col_y:
                st.markdown(f"""
                <div style="background: #eff6ff; padding: 25px; border-radius: 15px; border: 2px solid #3b82f6;">
                    <h3 style="color: #1e40af; margin: 0 0 15px 0; font-size: 18px;">📋 Next Steps</h3>
                    <p style="color: #1f2937; font-size: 14px; line-height: 1.8; background: white; padding: 15px; border-radius: 10px;">
                        {resolution['user_action']}
                    </p>
                    <div style="background: white; padding: 15px; border-radius: 10px; margin-top: 15px; text-align: center;">
                        <p style="color: #6b7280; font-size: 12px; margin: 0;">Similar tickets resolved</p>
                        <p style="color: #3b82f6; font-size: 28px; font-weight: 900; margin: 5px 0;">{classification['similar_tickets']}</p>
                        <p style="color: #6b7280; font-size: 11px; margin: 0;">Avg time: {classification['avg_resolution_time']}</p>
                    </div>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.warning("⚠️ This ticket requires human review and has been escalated to L2 support.")

with tab2:
    st.markdown("""
    <div style="background: linear-gradient(135deg, #f3e8ff 0%, #e9d5ff 100%); border: 3px solid #a855f7; border-radius: 16px; padding: 24px; margin-bottom: 20px;">
        <h3 style="color: #6b21a8; font-size: 22px; font-weight: 800; margin: 0;">IT Operations Dashboard</h3>
        <p style="color: #a855f7; font-size: 14px; margin: 8px 0 0 0;">Real-time metrics across your entire IT support operation</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Company metrics
    st.markdown("""
    <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); padding: 32px; border-radius: 20px; margin-bottom: 25px;">
        <h2 style="color: white; font-size: 28px; font-weight: 900; margin: 0 0 20px 0;">📊 This Month's Performance</h2>
        <div style="display: grid; grid-template-columns: repeat(4, 1fr); gap: 15px;">
            <div style="background: rgba(255,255,255,0.15); backdrop-filter: blur(10px); border-radius: 12px; padding: 20px; text-align: center;">
                <p style="font-size: 14px; color: rgba(255,255,255,0.8); margin: 0;">Tickets Resolved</p>
                <p style="font-size: 48px; color: white; font-weight: 900; margin: 8px 0;">1,847</p>
                <p style="font-size: 13px; color: rgba(255,255,255,0.7); margin: 0;">+23% vs last month</p>
            </div>
            <div style="background: rgba(255,255,255,0.15); backdrop-filter: blur(10px); border-radius: 12px; padding: 20px; text-align: center;">
                <p style="font-size: 14px; color: rgba(255,255,255,0.8); margin: 0;">Auto-Resolved</p>
                <p style="font-size: 48px; color: #86efac; font-weight: 900; margin: 8px 0;">72%</p>
                <p style="font-size: 13px; color: rgba(255,255,255,0.7); margin: 0;">No human needed</p>
            </div>
            <div style="background: rgba(255,255,255,0.15); backdrop-filter: blur(10px); border-radius: 12px; padding: 20px; text-align: center;">
                <p style="font-size: 14px; color: rgba(255,255,255,0.8); margin: 0;">Avg Resolution</p>
                <p style="font-size: 48px; color: white; font-weight: 900; margin: 8px 0;">4.2m</p>
                <p style="font-size: 13px; color: rgba(255,255,255,0.7); margin: 0;">vs 4.8hrs manual</p>
            </div>
            <div style="background: rgba(255,255,255,0.15); backdrop-filter: blur(10px); border-radius: 12px; padding: 20px; text-align: center;">
                <p style="font-size: 14px; color: rgba(255,255,255,0.8); margin: 0;">Cost Savings</p>
                <p style="font-size: 40px; color: #fbbf24; font-weight: 900; margin: 8px 0;">$105K</p>
                <p style="font-size: 13px; color: rgba(255,255,255,0.7); margin: 0;">This year</p>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Category breakdown
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        <div style="background: white; padding: 25px; border-radius: 15px; border: 2px solid #e5e7eb;">
            <h3 style="color: #1f2937; margin: 0 0 20px 0; font-size: 20px;">📋 Tickets by Category</h3>
            <div style="margin-bottom: 15px;">
                <div style="display: flex; justify-content: space-between; margin-bottom: 8px;">
                    <span style="color: #6b7280; font-size: 14px;">Password Resets</span>
                    <span style="color: #059669; font-weight: 700;">847 (46%)</span>
                </div>
                <div style="background: #e5e7eb; height: 8px; border-radius: 4px;">
                    <div style="background: #059669; width: 46%; height: 100%; border-radius: 4px;"></div>
                </div>
            </div>
            <div style="margin-bottom: 15px;">
                <div style="display: flex; justify-content: space-between; margin-bottom: 8px;">
                    <span style="color: #6b7280; font-size: 14px;">VPN/Network</span>
                    <span style="color: #3b82f6; font-weight: 700;">412 (22%)</span>
                </div>
                <div style="background: #e5e7eb; height: 8px; border-radius: 4px;">
                    <div style="background: #3b82f6; width: 22%; height: 100%; border-radius: 4px;"></div>
                </div>
            </div>
            <div style="margin-bottom: 15px;">
                <div style="display: flex; justify-content: space-between; margin-bottom: 8px;">
                    <span style="color: #6b7280; font-size: 14px;">Software Issues</span>
                    <span style="color: #8b5cf6; font-weight: 700;">318 (17%)</span>
                </div>
                <div style="background: #e5e7eb; height: 8px; border-radius: 4px;">
                    <div style="background: #8b5cf6; width: 17%; height: 100%; border-radius: 4px;"></div>
                </div>
            </div>
            <div>
                <div style="display: flex; justify-content: space-between; margin-bottom: 8px;">
                    <span style="color: #6b7280; font-size: 14px;">Hardware/Other</span>
                    <span style="color: #f59e0b; font-weight: 700;">270 (15%)</span>
                </div>
                <div style="background: #e5e7eb; height: 8px; border-radius: 4px;">
                    <div style="background: #f59e0b; width: 15%; height: 100%; border-radius: 4px;"></div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div style="background: white; padding: 25px; border-radius: 15px; border: 2px solid #e5e7eb;">
            <h3 style="color: #1f2937; margin: 0 0 20px 0; font-size: 20px;">⚡ Resolution Rate</h3>
            <div style="background: #ecfdf5; padding: 20px; border-radius: 10px; margin-bottom: 15px; text-align: center;">
                <p style="color: #047857; font-size: 14px; margin: 0;">Auto-Resolved</p>
                <p style="color: #059669; font-size: 48px; font-weight: 900; margin: 8px 0;">72%</p>
                <p style="color: #6b7280; font-size: 12px; margin: 0;">1,330 tickets</p>
            </div>
            <div style="background: #eff6ff; padding: 20px; border-radius: 10px; margin-bottom: 15px; text-align: center;">
                <p style="color: #1e40af; font-size: 14px; margin: 0;">AI-Assisted</p>
                <p style="color: #3b82f6; font-size: 48px; font-weight: 900; margin: 8px 0;">18%</p>
                <p style="color: #6b7280; font-size: 12px; margin: 0;">332 tickets</p>
            </div>
            <div style="background: #fef3c7; padding: 20px; border-radius: 10px; text-align: center;">
                <p style="color: #92400e; font-size: 14px; margin: 0;">Human Required</p>
                <p style="color: #f59e0b; font-size: 48px; font-weight: 900; margin: 8px 0;">10%</p>
                <p style="color: #6b7280; font-size: 12px; margin: 0;">185 tickets</p>
            </div>
        </div>
        """, unsafe_allow_html=True)

with tab3:
    st.markdown("""
    <div style="background: linear-gradient(135deg, #f0f9ff 0%, #e0f2fe 100%); border: 3px solid #3b82f6; border-radius: 16px; padding: 24px; margin-bottom: 20px;">
        <h3 style="color: #1e40af; font-size: 22px; font-weight: 800; margin: 0;">How Everest AI Works</h3>
        <p style="color: #3b82f6; font-size: 14px; margin: 8px 0 0 0;">Multi-step AI pipeline for intelligent ticket resolution</p>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        <div style="background: white; padding: 25px; border-radius: 15px; border: 2px solid #e5e7eb;">
            <h3 style="color: #1f2937; margin: 0 0 20px 0; font-size: 20px;">🔄 AI Pipeline</h3>
            <div style="background: #f9fafb; padding: 15px; border-radius: 10px; border-left: 4px solid #3b82f6; margin-bottom: 12px;">
                <h4 style="color: #1e40af; font-size: 16px; font-weight: 700; margin: 0 0 8px 0;">1. Intake & Classification</h4>
                <p style="color: #6b7280; font-size: 13px; margin: 0;">NLP analyzes ticket description, extracts key entities, classifies category/priority</p>
            </div>
            <div style="background: #f9fafb; padding: 15px; border-radius: 10px; border-left: 4px solid #10b981; margin-bottom: 12px;">
                <h4 style="color: #065f46; font-size: 16px; font-weight: 700; margin: 0 0 8px 0;">2. Knowledge Base Search</h4>
                <p style="color: #6b7280; font-size: 13px; margin: 0;">RAG system finds similar resolved tickets, pulls solutions from knowledge base</p>
            </div>
            <div style="background: #f9fafb; padding: 15px; border-radius: 10px; border-left: 4px solid #8b5cf6; margin-bottom: 12px;">
                <h4 style="color: #6b21a8; font-size: 16px; font-weight: 700; margin: 0 0 8px 0;">3. Automated Resolution</h4>
                <p style="color: #6b7280; font-size: 13px; margin: 0;">Execute fix via API integrations (AD, MDM, software deployment, VPN config)</p>
            </div>
            <div style="background: #f9fafb; padding: 15px; border-radius: 10px; border-left: 4px solid #f59e0b;">
                <h4 style="color: #92400e; font-size: 16px; font-weight: 700; margin: 0 0 8px 0;">4. Verification & Close</h4>
                <p style="color: #6b7280; font-size: 13px; margin: 0;">Confirm resolution, update user, close ticket, log for continuous learning</p>
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div style="background: white; padding: 25px; border-radius: 15px; border: 2px solid #e5e7eb;">
            <h3 style="color: #1f2937; margin: 0 0 20px 0; font-size: 20px;">🔌 System Integrations</h3>
            <div style="background: #ecfdf5; padding: 12px 15px; border-radius: 8px; margin-bottom: 10px;">
                <p style="color: #047857; font-weight: 700; font-size: 14px; margin: 0;">✓ Active Directory</p>
                <p style="color: #6b7280; font-size: 12px; margin: 3px 0 0 0;">Password resets, account unlocks, group membership</p>
            </div>
            <div style="background: #eff6ff; padding: 12px 15px; border-radius: 8px; margin-bottom: 10px;">
                <p style="color: #1e40af; font-weight: 700; font-size: 14px; margin: 0;">✓ MDM (Jamf/Intune)</p>
                <p style="color: #6b7280; font-size: 12px; margin: 3px 0 0 0;">Software deployment, remote configuration, device management</p>
            </div>
            <div style="background: #fef3c7; padding: 12px 15px; border-radius: 8px; margin-bottom: 10px;">
                <p style="color: #92400e; font-weight: 700; font-size: 14px; margin: 0;">✓ VPN/Network</p>
                <p style="color: #6b7280; font-size: 12px; margin: 3px 0 0 0;">Certificate refresh, config updates, connectivity checks</p>
            </div>
            <div style="background: #f3e8ff; padding: 12px 15px; border-radius: 8px; margin-bottom: 10px;">
                <p style="color: #6b21a8; font-weight: 700; font-size: 14px; margin: 0;">✓ Ticketing System</p>
                <p style="color: #6b7280; font-size: 12px; margin: 3px 0 0 0;">Jira, ServiceNow, Zendesk integration</p>
            </div>
            <div style="background: #fce7f3; padding: 12px 15px; border-radius: 8px;">
                <p style="color: #9f1239; font-weight: 700; font-size: 14px; margin: 0;">✓ Knowledge Base</p>
                <p style="color: #6b7280; font-size: 12px; margin: 3px 0 0 0;">RAG for solution retrieval, continuous learning</p>
            </div>
        </div>
        """, unsafe_allow_html=True)

# Footer
st.markdown("""
    <hr style="border: 3px solid #e5e7eb; margin: 45px 0; border-radius: 2px;">
    <div style="background: linear-gradient(135deg, #f9fafb 0%, #f3f4f6 100%); padding: 35px; border-radius: 20px; box-shadow: 0 8px 20px rgba(0,0,0,0.08); margin-bottom: 30px;">
        <h2 style="color: #3b82f6; margin: 0 0 25px 0; font-size: 32px; font-weight: 900; text-align: center;">🎯 Why This Matters for Everest</h2>
        <div style="display: grid; grid-template-columns: repeat(3, 1fr); gap: 20px; margin-bottom: 25px;">
            <div style="background: white; padding: 24px; border-radius: 16px; box-shadow: 0 4px 12px rgba(0,0,0,0.08); border-top: 5px solid #10b981;">
                <h4 style="color: #10b981; margin: 0 0 12px 0; font-size: 18px; font-weight: 800;">⚡ Instant Resolution</h4>
                <p style="color: #6b7280; font-size: 14px; line-height: 1.7; margin: 0;">
                    72% of tickets auto-resolve in <5 minutes. No queue, no waiting. Employees get help immediately, IT teams focus on complex issues.
                </p>
            </div>
            <div style="background: white; padding: 24px; border-radius: 16px; box-shadow: 0 4px 12px rgba(0,0,0,0.08); border-top: 5px solid #3b82f6;">
                <h4 style="color: #3b82f6; margin: 0 0 12px 0; font-size: 18px; font-weight: 800;">💰 Massive Savings</h4>
                <p style="color: #6b7280; font-size: 14px; line-height: 1.7; margin: 0;">
                    $105K annual savings per 500 employees. 70% reduction in outsourced IT costs. ROI in first 3 months.
                </p>
            </div>
            <div style="background: white; padding: 24px; border-radius: 16px; box-shadow: 0 4px 12px rgba(0,0,0,0.08); border-top: 5px solid #ec4899;">
                <h4 style="color: #ec4899; margin: 0 0 12px 0; font-size: 18px; font-weight: 800;">📈 Scales Effortlessly</h4>
                <p style="color: #6b7280; font-size: 14px; line-height: 1.7; margin: 0;">
                    Handle 1,000+ tickets/month with same quality. AI gets smarter with each ticket. Zero hiring as company grows.
                </p>
            </div>
        </div>
        <div style="background: linear-gradient(135deg, #dbeafe 0%, #bfdbfe 100%); border: 3px solid #3b82f6; border-radius: 16px; padding: 28px; margin-bottom: 25px;">
            <h3 style="color: #1e40af; margin: 0 0 18px 0; font-size: 24px; font-weight: 800;">💼 Enterprise Impact</h3>
            <ul style="margin: 0; padding-left: 28px; line-height: 2.2;">
                <li style="color: #1f2937; font-size: 15px; font-weight: 600;"><strong style="color: #3b82f6;">72% auto-resolution:</strong> No human intervention needed</li>
                <li style="color: #1f2937; font-size: 15px; font-weight: 600;"><strong style="color: #3b82f6;">4.2 min avg resolution:</strong> vs 4.8 hours manual</li>
                <li style="color: #1f2937; font-size: 15px; font-weight: 600;"><strong style="color: #3b82f6;">$105K savings:</strong> Annual cost reduction per 500 employees</li>
                <li style="color: #1f2937; font-size: 15px; font-weight: 600;"><strong style="color: #3b82f6;">24/7 availability:</strong> No after-hours wait times</li>
            </ul>
        </div>
        <div style="background: linear-gradient(135deg, #d1fae5 0%, #a7f3d0 100%); border: 3px solid #10b981; border-radius: 16px; padding: 28px;">
            <h3 style="color: #065f46; margin: 0 0 18px 0; font-size: 24px; font-weight: 800;">⚡ Technical Stack</h3>
            <div style="display: grid; grid-template-columns: repeat(2, 1fr); gap: 15px;">
                <div style="background: white; border-radius: 12px; padding: 18px;">
                    <p style="font-size: 14px; color: #059669; font-weight: 700; margin: 0 0 6px 0;">✓ NLP Classification</p>
                    <p style="font-size: 13px; color: #6b7280; margin: 0;">Ticket categorization, intent detection, entity extraction</p>
                </div>
                <div style="background: white; border-radius: 12px; padding: 18px;">
                    <p style="font-size: 14px; color: #059669; font-weight: 700; margin: 0 0 6px 0;">✓ RAG System</p>
                    <p style="font-size: 13px; color: #6b7280; margin: 0;">Knowledge base search, similar ticket retrieval</p>
                </div>
                <div style="background: white; border-radius: 12px; padding: 18px;">
                    <p style="font-size: 14px; color: #059669; font-weight: 700; margin: 0 0 6px 0;">✓ API Integrations</p>
                    <p style="font-size: 13px; color: #6b7280; margin: 0;">AD, MDM, VPN, ticketing systems</p>
                </div>
                <div style="background: white; border-radius: 12px; padding: 18px;">
                    <p style="font-size: 14px; color: #059669; font-weight: 700; margin: 0 0 6px 0;">✓ Automation Engine</p>
                    <p style="font-size: 13px; color: #6b7280; margin: 0;">Workflow orchestration, script execution</p>
                </div>
            </div>
        </div>
    </div>
    <div style="text-align: center; padding: 40px; background: linear-gradient(135deg, #1e3a8a 0%, #3b82f6 100%); border-radius: 20px; box-shadow: 0 12px 28px rgba(59, 130, 246, 0.35); color: white;">
        <h3 style="margin: 0 0 18px 0; font-size: 28px; font-weight: 900;">👨‍💻 About This Demo</h3>
        <p style="font-size: 18px; margin: 12px 0; font-weight: 600;">
            Built for <strong style="color: white;">Everest</strong> by <strong style="color: white;">Anju Vilashni Nandhakumar</strong>
        </p>
        <div style="margin: 25px 0; padding: 22px; background: rgba(255,255,255,0.15); backdrop-filter: blur(10px); border-radius: 16px; border: 1px solid rgba(255,255,255,0.2);">
            <p style="margin: 8px 0; font-size: 16px;">
                📧 <a href="mailto:nandhakumar.anju@gmail.com" style="color: white; font-weight: 700; text-decoration: none;">nandhakumar.anju@gmail.com</a>
            </p>
            <p style="margin: 8px 0; font-size: 16px;">
                💼 <a href="https://linkedin.com/in/anju-vilashni" target="_blank" style="color: white; font-weight: 700; text-decoration: none;">LinkedIn</a> | 
                💻 <a href="https://github.com/Av1352" target="_blank" style="color: white; font-weight: 700; text-decoration: none;">GitHub</a> | 
                🌐 <a href="https://vxanju.com" target="_blank" style="color: white; font-weight: 700; text-decoration: none;">Portfolio</a>
            </p>
        </div>
        <p style="font-size: 15px; margin: 18px 0; font-weight: 700;">
            <strong style="color: white;">Tech Stack:</strong> NLP • RAG • Automation • API Integration • Agentic Systems
        </p>
        <hr style="border: 1px solid rgba(255,255,255,0.3); margin: 25px 0;">
        <p style="font-size: 14px; font-style: italic; line-height: 1.8; max-width: 900px; margin: 0 auto; color: rgba(255,255,255,0.9);">
            Demo showcasing AI-powered IT support automation with intelligent ticket resolution.<br>
            NLP classification • RAG knowledge base • Automated fixes • System integrations • Analytics
        </p>
    </div>
    """, unsafe_allow_html=True)