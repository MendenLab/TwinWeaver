// Remove emojis from Table of Contents (TOC) links while keeping them in main content
document.addEventListener('DOMContentLoaded', function() {
    removeEmojisFromTOC();
});

// Also run on navigation for single-page app behavior
document.addEventListener('DOMContentLoaded', function() {
    // For instant navigation in Material theme
    if (typeof document$ !== 'undefined') {
        document$.subscribe(function() {
            removeEmojisFromTOC();
        });
    }
});

function removeEmojisFromTOC() {
    // Select all TOC links in the right sidebar (secondary navigation)
    const tocLinks = document.querySelectorAll('.md-sidebar--secondary .md-nav__link, .md-nav--secondary .md-nav__link');

    // Emoji regex pattern - covers most common emoji ranges
    const emojiRegex = /[\u{1F300}-\u{1F9FF}]|[\u{2600}-\u{26FF}]|[\u{2700}-\u{27BF}]|[\u{1F600}-\u{1F64F}]|[\u{1F680}-\u{1F6FF}]|[\u{1F1E0}-\u{1F1FF}]|[\u{1F900}-\u{1F9FF}]|[\u{1FA00}-\u{1FA6F}]|[\u{1FA70}-\u{1FAFF}]|[\u{231A}-\u{231B}]|[\u{23E9}-\u{23F3}]|[\u{23F8}-\u{23FA}]|[\u{25AA}-\u{25AB}]|[\u{25B6}]|[\u{25C0}]|[\u{25FB}-\u{25FE}]|[\u{2614}-\u{2615}]|[\u{2648}-\u{2653}]|[\u{267F}]|[\u{2693}]|[\u{26A1}]|[\u{26AA}-\u{26AB}]|[\u{26BD}-\u{26BE}]|[\u{26C4}-\u{26C5}]|[\u{26CE}]|[\u{26D4}]|[\u{26EA}]|[\u{26F2}-\u{26F3}]|[\u{26F5}]|[\u{26FA}]|[\u{26FD}]|[\u{2702}]|[\u{2705}]|[\u{2708}-\u{270D}]|[\u{270F}]|[\u{2712}]|[\u{2714}]|[\u{2716}]|[\u{271D}]|[\u{2721}]|[\u{2728}]|[\u{2733}-\u{2734}]|[\u{2744}]|[\u{2747}]|[\u{274C}]|[\u{274E}]|[\u{2753}-\u{2755}]|[\u{2757}]|[\u{2763}-\u{2764}]|[\u{2795}-\u{2797}]|[\u{27A1}]|[\u{27B0}]|[\u{27BF}]|[\u{2934}-\u{2935}]|[\u{2B05}-\u{2B07}]|[\u{2B1B}-\u{2B1C}]|[\u{2B50}]|[\u{2B55}]|[\u{3030}]|[\u{303D}]|[\u{3297}]|[\u{3299}]|[\u{FE0F}]/gu;

    tocLinks.forEach(function(link) {
        // Get text nodes only (preserve any child elements)
        const walker = document.createTreeWalker(link, NodeFilter.SHOW_TEXT, null, false);
        let node;
        while (node = walker.nextNode()) {
            if (emojiRegex.test(node.textContent)) {
                node.textContent = node.textContent.replace(emojiRegex, '').trim();
            }
        }
    });
}
