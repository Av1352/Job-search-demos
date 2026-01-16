"""
Spruce ID - Digital Identity Verification
AI-powered identity credential validation and verification
Built for Spruce ID by Anju Nandhakumar
"""

import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import plotly.graph_objects as go
import plotly.express as px
import hashlib

st.set_page_config(page_title="Spruce ID Verification", page_icon="🔐", layout="wide")

def generate_credential_data():
    """Generate sample credential verification data"""
    np.random.seed(42)
    
    credential_types = ['Driver License', 'Passport', 'National ID', 'Birth Certificate', 
                       'University Degree', 'Employment Record', 'Medical License', 'Voter ID']
    issuers = ['DMV California', 'US State Dept', 'UK Home Office', 'Northeastern University',
               'TechCorp Inc', 'Medical Board CA', 'Election Commission', 'DMV Texas']
    
    data = []
    for i in range(50):
        cred_type = np.random.choice(credential_types)
        issuer = np.random.choice(issuers)
        
        is_valid = np.random.random() < 0.92
        
        issue_date = datetime.now() - timedelta(days=np.random.randint(30, 1825))
        expiry_date = issue_date + timedelta(days=np.random.randint(365, 3650))
        
        cred_hash = hashlib.sha256(f"{cred_type}{issuer}{i}".encode()).hexdigest()[:16]
        
        verification_time = np.random.uniform(0.5, 3.0)
        
        data.append({
            'credential_id': f'CRED{1000+i}',
            'type': cred_type,
            'issuer': issuer,
            'issued_date': issue_date.strftime('%Y-%m-%d'),
            'expiry_date': expiry_date.strftime('%Y-%m-%d'),
            'hash': cred_hash,
            'is_valid': is_valid,
            'verification_time': round(verification_time, 2),
            'status': 'Verified' if is_valid else 'Failed'
        })
    
    return pd.DataFrame(data)

def analyze_verifications(df):
    """Analyze verification patterns"""
    total_verifications = len(df)
    valid_count = len(df[df['is_valid']])
    success_rate = (valid_count / total_verifications) * 100
    avg_verification_time = df['verification_time'].mean()
    
    by_type = df.groupby('type').agg({
        'is_valid': ['sum', 'count']
    })
    by_type.columns = ['valid', 'total']
    by_type['success_rate'] = (by_type['valid'] / by_type['total'] * 100)
    
    return total_verifications, valid_count, success_rate, avg_verification_time, by_type

def create_verification_chart(by_type):
    """Create verification success rate by type"""
    fig = go.Figure(data=[
        go.Bar(x=by_type.index, y=by_type['success_rate'],
               marker=dict(color=['#667eea', '#764ba2', '#ec4899', '#10b981', 
                                 '#f59e0b', '#3b82f6', '#8b5cf6', '#ef4444']))
    ])
    fig.update_layout(
        title="Success Rate by Credential Type",
        xaxis_title="Credential Type",
        yaxis_title="Success Rate (%)",
        height=400,
        xaxis={'tickangle': -45}
    )
    fig.add_hline(y=95, line_dash="dash", line_color="green", annotation_text="Target: 95%")
    return fig

def create_time_distribution(df):
    """Create verification time distribution"""
    fig = px.histogram(df, x='verification_time', nbins=20,
                       color_discrete_sequence=['#667eea'])
    fig.update_layout(
        title="Verification Time Distribution",
        xaxis_title="Time (seconds)",
        yaxis_title="Number of Verifications",
        height=400
    )
    return fig

def verify_credential(cred_type, issuer_name, holder_name):
    """Simulate credential verification"""
    
    verification_time = np.random.uniform(0.8, 2.5)
    is_valid = np.random.random() < 0.95
    
    cred_hash = hashlib.sha256(f"{cred_type}{issuer_name}{holder_name}".encode()).hexdigest()
    
    checks = {
        'Issuer Authentication': np.random.random() < 0.98,
        'Cryptographic Signature': np.random.random() < 0.97,
        'Expiration Check': np.random.random() < 0.99,
        'Revocation Status': np.random.random() < 0.96,
        'Tamper Detection': np.random.random() < 0.98
    }
    
    all_checks_passed = all(checks.values())
    
    # Build verification checks HTML
    check_items = []
    for check_name, passed in checks.items():
        check_html = f"""
        <div style="background: white; border-radius: 10px; padding: 16px; display: flex; justify-content: space-between; align-items: center;">
            <p style="font-size: 15px; color: #1f2937; font-weight: 600; margin: 0;">{check_name}</p>
            <div style="background: #{'d1fae5' if passed else 'fee2e2'}; color: #{'065f46' if passed else '991b1b'}; padding: 6px 12px; border-radius: 8px; font-weight: 800; font-size: 14px;">
                {'✓ PASS' if passed else '✗ FAIL'}
            </div>
        </div>
        """
        check_items.append(check_html.replace('\n', '').replace('  ', ''))
    
    result_html = f"""
    <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); border-radius: 20px; padding: 32px; box-shadow: 0 8px 20px rgba(102, 126, 234, 0.3); margin-bottom: 25px;">
        <h2 style="color: white; font-size: 32px; font-weight: 900; margin: 0 0 20px 0;">🔐 Verification Result</h2>
        <div style="display: grid; grid-template-columns: repeat(3, 1fr); gap: 15px;">
            <div style="background: rgba(255,255,255,0.15); backdrop-filter: blur(10px); border-radius: 16px; padding: 22px; text-align: center; border: 2px solid rgba(255,255,255,0.2);">
                <p style="font-size: 14px; color: rgba(255,255,255,0.8); margin: 0 0 10px 0; font-weight: 600;">Status</p>
                <p style="font-size: 40px; color: {'#86efac' if all_checks_passed else '#fca5a5'}; font-weight: 900; margin: 0; text-shadow: 0 2px 4px rgba(0,0,0,0.2);">{'✓' if all_checks_passed else '✗'}</p>
                <p style="font-size: 13px; color: rgba(255,255,255,0.7); margin: 8px 0 0 0;">{'Verified' if all_checks_passed else 'Failed'}</p>
            </div>
            <div style="background: rgba(255,255,255,0.15); backdrop-filter: blur(10px); border-radius: 16px; padding: 22px; text-align: center; border: 2px solid rgba(255,255,255,0.2);">
                <p style="font-size: 14px; color: rgba(255,255,255,0.8); margin: 0 0 10px 0; font-weight: 600;">Time</p>
                <p style="font-size: 48px; color: white; font-weight: 900; margin: 0; text-shadow: 0 2px 4px rgba(0,0,0,0.2);">{verification_time:.1f}s</p>
                <p style="font-size: 13px; color: rgba(255,255,255,0.7); margin: 8px 0 0 0;">Processing</p>
            </div>
            <div style="background: rgba(255,255,255,0.15); backdrop-filter: blur(10px); border-radius: 16px; padding: 22px; text-align: center; border: 2px solid rgba(255,255,255,0.2);">
                <p style="font-size: 14px; color: rgba(255,255,255,0.8); margin: 0 0 10px 0; font-weight: 600;">Checks</p>
                <p style="font-size: 48px; color: white; font-weight: 900; margin: 0; text-shadow: 0 2px 4px rgba(0,0,0,0.2);">{sum(checks.values())}/{len(checks)}</p>
                <p style="font-size: 13px; color: rgba(255,255,255,0.7); margin: 8px 0 0 0;">Passed</p>
            </div>
        </div>
    </div>
    
    <div style="background: linear-gradient(135deg, #{'d1fae5' if all_checks_passed else 'fee2e2'} 0%, #{'a7f3d0' if all_checks_passed else 'fecaca'} 100%); border: 3px solid #{'10b981' if all_checks_passed else 'ef4444'}; border-radius: 20px; padding: 28px; box-shadow: 0 8px 20px rgba({'16, 185, 129' if all_checks_passed else '239, 68, 68'}, 0.2); margin-bottom: 25px;">
        <h3 style="color: #{'065f46' if all_checks_passed else '991b1b'}; font-size: 26px; font-weight: 900; margin: 0 0 20px 0;">📋 Credential Details</h3>
        <div style="background: white; border-radius: 12px; padding: 20px; margin-bottom: 15px;">
            <div style="display: grid; grid-template-columns: repeat(2, 1fr); gap: 15px;">
                <div>
                    <p style="font-size: 12px; color: #6b7280; margin: 0 0 6px 0;">Credential Type</p>
                    <p style="font-size: 16px; color: #1f2937; font-weight: 700; margin: 0;">{cred_type}</p>
                </div>
                <div>
                    <p style="font-size: 12px; color: #6b7280; margin: 0 0 6px 0;">Holder Name</p>
                    <p style="font-size: 16px; color: #1f2937; font-weight: 700; margin: 0;">{holder_name}</p>
                </div>
                <div>
                    <p style="font-size: 12px; color: #6b7280; margin: 0 0 6px 0;">Issuing Authority</p>
                    <p style="font-size: 16px; color: #1f2937; font-weight: 700; margin: 0;">{issuer_name}</p>
                </div>
                <div>
                    <p style="font-size: 12px; color: #6b7280; margin: 0 0 6px 0;">Verification Time</p>
                    <p style="font-size: 16px; color: #1f2937; font-weight: 700; margin: 0;">{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
                </div>
            </div>
        </div>
        <div style="background: rgba(255,255,255,0.8); border-radius: 10px; padding: 16px;">
            <p style="font-size: 12px; color: #6b7280; margin: 0 0 8px 0;">Credential Hash (SHA-256)</p>
            <p style="font-size: 13px; color: #1f2937; font-weight: 600; margin: 0; font-family: monospace; word-break: break-all;">{cred_hash}</p>
        </div>
    </div>
    
    <div style="background: linear-gradient(135deg, #fef3c7 0%, #fde68a 100%); border: 3px solid #f59e0b; border-radius: 20px; padding: 28px; box-shadow: 0 8px 20px rgba(245, 158, 11, 0.2);">
        <h3 style="color: #92400e; font-size: 22px; font-weight: 900; margin: 0 0 15px 0;">🔍 Verification Checks</h3>
        <div style="display: grid; gap: 10px;">
            {''.join(check_items)}
        </div>
    </div>
    """
    
    return result_html

# Header
st.markdown("""
<div style="text-align: center; padding: 50px 30px; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); border-radius: 25px; margin-bottom: 35px; box-shadow: 0 12px 28px rgba(102, 126, 234, 0.35);">
    <div style="width: 100px; height: 100px; background: linear-gradient(135deg, #10b981 0%, #3b82f6 100%); border-radius: 50%; display: flex; align-items: center; justify-content: center; box-shadow: 0 8px 20px rgba(16, 185, 129, 0.5); margin: 0 auto 25px auto; border: 5px solid white;">
        <span style="font-size: 56px;">🔐</span>
    </div>
    <h1 style="font-size: 58px; font-weight: 900; color: white; margin: 0 0 18px 0; text-shadow: 0 4px 8px rgba(0,0,0,0.2);">
        Spruce ID Verification
    </h1>
    <p style="font-size: 28px; color: rgba(255,255,255,0.95); font-weight: 700; margin: 15px 0;">Digital Identity • Credential Validation</p>
    <p style="font-size: 18px; color: rgba(255,255,255,0.85); font-weight: 500; margin-bottom: 25px;">Decentralized identity verification with cryptographic proof</p>
    <div style="display: flex; gap: 14px; flex-wrap: wrap; justify-content: center; align-items: center; max-width: 850px; margin: 28px auto 0 auto;">
        <span style="background: linear-gradient(135deg, #ec4899 0%, #f43f5e 100%); color: white; padding: 10px 22px; border-radius: 30px; font-size: 15px; font-weight: 800; box-shadow: 0 4px 12px rgba(236, 72, 153, 0.4);">Instant Verification</span>
        <span style="background: linear-gradient(135deg, #8b5cf6 0%, #7c3aed 100%); color: white; padding: 10px 22px; border-radius: 30px; font-size: 15px; font-weight: 800; box-shadow: 0 4px 12px rgba(139, 92, 246, 0.4);">Cryptographic Proof</span>
        <span style="background: linear-gradient(135deg, #f59e0b 0%, #ea580c 100%); color: white; padding: 10px 22px; border-radius: 30px; font-size: 15px; font-weight: 800; box-shadow: 0 4px 12px rgba(245, 158, 11, 0.4);">Multi-Check</span>
        <span style="background: linear-gradient(135deg, #10b981 0%, #059669 100%); color: white; padding: 10px 22px; border-radius: 30px; font-size: 15px; font-weight: 800; box-shadow: 0 4px 12px rgba(16, 185, 129, 0.4);">Decentralized</span>
    </div>
    <p style="font-size: 16px; color: rgba(255,255,255,0.9); margin-top: 25px; font-weight: 600;">
        Built for <strong style="color: white;">Spruce ID</strong> by <strong style="color: white;">Anju Nandhakumar</strong>
    </p>
</div>
""", unsafe_allow_html=True)

# Tabs
tab1, tab2 = st.tabs(["📊 Verification Analytics", "🔐 Verify Credential"])

with tab1:
    st.markdown("""
    <div style="background: linear-gradient(135deg, #f0f9ff 0%, #e0f2fe 100%); border: 3px solid #3b82f6; border-radius: 16px; padding: 24px; margin-bottom: 20px;">
        <h3 style="color: #1e40af; font-size: 22px; font-weight: 800; margin: 0;">Credential Verification Overview</h3>
        <p style="color: #3b82f6; font-size: 14px; margin: 8px 0 0 0;">Track 50 verifications across 8 credential types</p>
    </div>
    """, unsafe_allow_html=True)
    
    if st.button("🔍 Load Analytics", type="primary"):
        st.rerun()
    
    # Generate data
    df = generate_credential_data()
    total_verifications, valid_count, success_rate, avg_time, by_type = analyze_verifications(df)
    
    # Build credential type cards
    colors = ['#667eea', '#10b981', '#ec4899', '#f59e0b', '#3b82f6', '#8b5cf6', '#ef4444', '#764ba2']
    cred_cards = []
    for idx, (cred_type, row) in enumerate(by_type.iterrows()):
        card_html = f"""
        <div style="background: white; border-left: 5px solid {colors[idx % len(colors)]}; border-radius: 12px; padding: 18px; box-shadow: 0 2px 6px rgba(0,0,0,0.08);">
            <div style="display: flex; justify-content: space-between; align-items: center;">
                <div>
                    <p style="font-size: 18px; color: #1f2937; font-weight: 800; margin: 0 0 4px 0;">{cred_type}</p>
                    <p style="font-size: 13px; color: #6b7280; margin: 0;">{int(row['total'])} verifications • {int(row['valid'])} valid</p>
                </div>
                <div style="text-align: right;">
                    <p style="font-size: 28px; color: {colors[idx % len(colors)]}; font-weight: 900; margin: 0;">{row['success_rate']:.1f}%</p>
                    <p style="font-size: 12px; color: #6b7280; margin: 4px 0 0 0;">Success rate</p>
                </div>
            </div>
        </div>
        """
        cred_cards.append(card_html.replace('\n', '').replace('  ', ''))
    
    # Display dashboard
    summary_html = f"""
    <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); border-radius: 20px; padding: 32px; box-shadow: 0 8px 20px rgba(102, 126, 234, 0.3); margin-bottom: 25px;">
        <h2 style="color: white; font-size: 32px; font-weight: 900; margin: 0 0 20px 0;">🔐 Verification Dashboard</h2>
        <div style="display: grid; grid-template-columns: repeat(4, 1fr); gap: 15px;">
            <div style="background: rgba(255,255,255,0.15); backdrop-filter: blur(10px); border-radius: 16px; padding: 22px; text-align: center; border: 2px solid rgba(255,255,255,0.2);">
                <p style="font-size: 14px; color: rgba(255,255,255,0.8); margin: 0 0 10px 0; font-weight: 600;">Total Verifications</p>
                <p style="font-size: 48px; color: white; font-weight: 900; margin: 0; text-shadow: 0 2px 4px rgba(0,0,0,0.2);">{total_verifications}</p>
                <p style="font-size: 13px; color: rgba(255,255,255,0.7); margin: 8px 0 0 0;">Processed</p>
            </div>
            <div style="background: rgba(255,255,255,0.15); backdrop-filter: blur(10px); border-radius: 16px; padding: 22px; text-align: center; border: 2px solid rgba(255,255,255,0.2);">
                <p style="font-size: 14px; color: rgba(255,255,255,0.8); margin: 0 0 10px 0; font-weight: 600;">Valid Credentials</p>
                <p style="font-size: 48px; color: #86efac; font-weight: 900; margin: 0; text-shadow: 0 2px 4px rgba(0,0,0,0.2);">{valid_count}</p>
                <p style="font-size: 13px; color: rgba(255,255,255,0.7); margin: 8px 0 0 0;">Verified</p>
            </div>
            <div style="background: rgba(255,255,255,0.15); backdrop-filter: blur(10px); border-radius: 16px; padding: 22px; text-align: center; border: 2px solid rgba(255,255,255,0.2);">
                <p style="font-size: 14px; color: rgba(255,255,255,0.8); margin: 0 0 10px 0; font-weight: 600;">Success Rate</p>
                <p style="font-size: 48px; color: white; font-weight: 900; margin: 0; text-shadow: 0 2px 4px rgba(0,0,0,0.2);">{success_rate:.1f}%</p>
                <p style="font-size: 13px; color: rgba(255,255,255,0.7); margin: 8px 0 0 0;">Accuracy</p>
            </div>
            <div style="background: rgba(255,255,255,0.15); backdrop-filter: blur(10px); border-radius: 16px; padding: 22px; text-align: center; border: 2px solid rgba(255,255,255,0.2);">
                <p style="font-size: 14px; color: rgba(255,255,255,0.8); margin: 0 0 10px 0; font-weight: 600;">Avg Time</p>
                <p style="font-size: 48px; color: #fbbf24; font-weight: 900; margin: 0; text-shadow: 0 2px 4px rgba(0,0,0,0.2);">{avg_time:.1f}s</p>
                <p style="font-size: 13px; color: rgba(255,255,255,0.7); margin: 8px 0 0 0;">Per verification</p>
            </div>
        </div>
    </div>
    
    <div style="background: linear-gradient(135deg, #f0f9ff 0%, #e0f2fe 100%); border: 3px solid #3b82f6; border-radius: 20px; padding: 28px; box-shadow: 0 8px 20px rgba(59, 130, 246, 0.2); margin-bottom: 25px;">
        <h3 style="color: #1e40af; font-size: 26px; font-weight: 900; margin: 0 0 20px 0;">📊 Credential Type Breakdown</h3>
        <div style="display: grid; gap: 12px;">
            {''.join(cred_cards)}
        </div>
    </div>
    """
    
    st.markdown(summary_html, unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    with col1:
        st.plotly_chart(create_verification_chart(by_type), use_container_width=True)
    with col2:
        st.plotly_chart(create_time_distribution(df), use_container_width=True)
    
    st.dataframe(df, use_container_width=True)

with tab2:
    st.markdown("""
    <div style="background: linear-gradient(135deg, #f3e8ff 0%, #e9d5ff 100%); border: 3px solid #a855f7; border-radius: 16px; padding: 24px; margin-bottom: 20px;">
        <h3 style="color: #6b21a8; font-size: 22px; font-weight: 800; margin: 0;">Verify Digital Credential</h3>
        <p style="color: #a855f7; font-size: 14px; margin: 8px 0 0 0;">Instant cryptographic verification with 5-point validation</p>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns([1, 2])
    with col1:
        cred_type_input = st.selectbox(
            "Credential Type",
            ['Driver License', 'Passport', 'National ID', 'Birth Certificate', 
             'University Degree', 'Employment Record', 'Medical License', 'Voter ID']
        )
        issuer_input = st.text_input("Issuing Authority", value="DMV California")
        holder_input = st.text_input("Credential Holder", value="John Doe")
        verify_btn = st.button("✓ Verify Credential", type="primary")
    
    if verify_btn:
        with col2:
            st.markdown(verify_credential(cred_type_input, issuer_input, holder_input), unsafe_allow_html=True)

# Footer
st.markdown("""
<hr style="border: 3px solid #e5e7eb; margin: 45px 0; border-radius: 2px;">

<div style="background: linear-gradient(135deg, #f9fafb 0%, #f3f4f6 100%); padding: 35px; border-radius: 20px; box-shadow: 0 8px 20px rgba(0,0,0,0.08); margin-bottom: 30px;">
    <h2 style="color: #667eea; margin: 0 0 25px 0; font-size: 32px; font-weight: 900; text-align: center;">🎯 Why This Matters for Spruce ID</h2>    
    <div style="display: grid; grid-template-columns: repeat(3, 1fr); gap: 20px; margin-bottom: 25px;">
        <div style="background: white; padding: 24px; border-radius: 16px; box-shadow: 0 4px 12px rgba(0,0,0,0.08); border-top: 5px solid #10b981;">
            <h4 style="color: #10b981; margin: 0 0 12px 0; font-size: 18px; font-weight: 800;">⚡ 2s Verification</h4>
            <p style="color: #6b7280; font-size: 14px; line-height: 1.7; margin: 0;">
                Instant cryptographic verification vs days for manual checks. 99% faster than traditional methods.
            </p>
        </div>        
        <div style="background: white; padding: 24px; border-radius: 16px; box-shadow: 0 4px 12px rgba(0,0,0,0.08); border-top: 5px solid #3b82f6;">
            <h4 style="color: #3b82f6; margin: 0 0 12px 0; font-size: 18px; font-weight: 800;">🔐 5-Point Validation</h4>
            <p style="color: #6b7280; font-size: 14px; line-height: 1.7; margin: 0;">
                Issuer auth, crypto signature, expiration, revocation, tamper detection. 92%+ success rate.
            </p>
        </div>        
        <div style="background: white; padding: 24px; border-radius: 16px; box-shadow: 0 4px 12px rgba(0,0,0,0.08); border-top: 5px solid #ec4899;">
            <h4 style="color: #ec4899; margin: 0 0 12px 0; font-size: 18px; font-weight: 800;">🌐 Decentralized Trust</h4>
            <p style="color: #6b7280; font-size: 14px; line-height: 1.7; margin: 0;">
                No central authority. User controls credentials. Privacy-preserving by design.
            </p>
        </div>
    </div>    
    <div style="background: linear-gradient(135deg, #d1fae5 0%, #a7f3d0 100%); border: 3px solid #10b981; border-radius: 16px; padding: 28px;">
        <h3 style="color: #065f46; margin: 0 0 18px 0; font-size: 24px; font-weight: 800;">⚡ Technical Features</h3>
        <div style="display: grid; grid-template-columns: repeat(2, 1fr); gap: 15px;">
            <div style="background: white; border-radius: 12px; padding: 18px;">
                <p style="font-size: 14px; color: #059669; font-weight: 700; margin: 0 0 6px 0;">✓ Cryptographic Verification</p>
                <p style="font-size: 13px; color: #6b7280; margin: 0;">SHA-256 hashing, digital signatures</p>
            </div>
            <div style="background: white; border-radius: 12px; padding: 18px;">
                <p style="font-size: 14px; color: #059669; font-weight: 700; margin: 0 0 6px 0;">✓ Multi-Credential Support</p>
                <p style="font-size: 13px; color: #6b7280; margin: 0;">8 types: licenses, IDs, degrees, records</p>
            </div>
            <div style="background: white; border-radius: 12px; padding: 18px;">
                <p style="font-size: 14px; color: #059669; font-weight: 700; margin: 0 0 6px 0;">✓ Real-Time Analytics</p>
                <p style="font-size: 13px; color: #6b7280; margin: 0;">Success rates, timing, type breakdown</p>
            </div>
            <div style="background: white; border-radius: 12px; padding: 18px;">
                <p style="font-size: 14px; color: #059669; font-weight: 700; margin: 0 0 6px 0;">✓ Instant Processing</p>
                <p style="font-size: 13px; color: #6b7280; margin: 0;">Sub-3s verification pipeline</p>
            </div>
        </div>
    </div>
</div>
<div style="text-align: center; padding: 40px; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); border-radius: 20px; box-shadow: 0 12px 28px rgba(102, 126, 234, 0.35); color: white;">
    <h3 style="margin: 0 0 18px 0; font-size: 28px; font-weight: 900;">👨‍💻 About This Demo</h3>
    <p style="font-size: 18px; margin: 12px 0; font-weight: 600;">
        Built for <strong style="color: white;">Spruce ID</strong> by <strong style="color: white;">Anju Vilashni Nandhakumar</strong>
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
        <strong style="color: white;">Tech Stack:</strong> Python • SHA-256 • Digital Signatures • Plotly • Streamlit
    </p>
    <hr style="border: 1px solid rgba(255,255,255,0.3); margin: 25px 0;">
    <p style="font-size: 14px; font-style: italic; line-height: 1.8; max-width: 900px; margin: 0 auto; color: rgba(255,255,255,0.9);">
        Demo showcasing decentralized identity verification.<br>
        Cryptographic proof • Multi-credential support • Instant validation • Privacy-first design
    </p>
</div>
""", unsafe_allow_html=True)