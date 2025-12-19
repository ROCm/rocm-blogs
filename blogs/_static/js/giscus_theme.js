
document.addEventListener("DOMContentLoaded", function () {
    const giscusOrigin = "https://giscus.app";

    function updateGiscusTheme() {
        const theme = document.documentElement.getAttribute("data-theme") || "light";
        const giscusTheme = theme === "dark" ? "transparent_dark" : "light";

        const iframe = document.querySelector("iframe.giscus-frame");
        if (iframe && iframe.contentWindow) {
            iframe.contentWindow.postMessage(
                {
                    giscus: {
                        setConfig: {
                            theme: giscusTheme,
                        },
                    },
                },
                giscusOrigin
            );
        }
    }

    // Watch for theme changes on the <html> element
    const observer = new MutationObserver(function (mutations) {
        mutations.forEach(function (mutation) {
            if (mutation.type === "attributes" && mutation.attributeName === "data-theme") {
                updateGiscusTheme();
            }
        });
    });

    observer.observe(document.documentElement, {
        attributes: true,
        attributeFilter: ["data-theme"],
    });

    // Listen for Giscus load to ensure we sync the correct theme initially
    window.addEventListener("message", function (event) {
        if (event.origin === giscusOrigin) {
            if (!(event.data && event.data.giscus && event.data.giscus.error)) {
                updateGiscusTheme();
            }
        }
    });
});
