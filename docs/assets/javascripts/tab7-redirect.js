(function () {
  "use strict";

  function redirectLegacyExplorerPage() {
    var link = document.getElementById("tab7-guide-redirect");
    if (!link) return;
    var target = new URL(link.href, document.baseURI);
    if (window.location.hash) target.hash = window.location.hash;
    window.location.replace(target.href);
  }

  // Material instant navigation replaces page content without reloading scripts.
  if (typeof document$ !== "undefined") {
    document$.subscribe(redirectLegacyExplorerPage);
  } else if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", redirectLegacyExplorerPage);
  } else {
    redirectLegacyExplorerPage();
  }
})();
