// src/components/Header.js

import React from 'react';

function Header() {
    return (
        <header className="header">
            <div className="logo">Dashboard</div>
            <nav>
                <ul className="nav-list">
                    <li><a href="/">Home</a></li>
                    <li><a href="/features">Features</a></li>
                    <li><a href="/contact">Contact</a></li>
                </ul>
            </nav>
        </header>
    );
}

export default Header;
